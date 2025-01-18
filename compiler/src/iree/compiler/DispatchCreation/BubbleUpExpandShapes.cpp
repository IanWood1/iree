// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//===--- BubbleExpandShapes.cpp --- Pass to propagate expand shapes op up -===//
//
// This pass propagates expand_shape operations up the program (and conversely)
// sinks the collapse_shape operations down the program to get the elementwise
// operations into higher dimensionality to get better fusion.
//
//===----------------------------------------------------------------------===//

#include "iree/compiler/Dialect/Flow/Transforms/RegionOpUtils.h"
#include "iree/compiler/Dialect/LinalgExt/IR/LinalgExtOps.h"
#include "iree/compiler/Dialect/LinalgExt/Transforms/Transforms.h"
#include "iree/compiler/Dialect/LinalgExt/Utils/IndexingUtils.h"
#include "iree/compiler/Dialect/LinalgExt/Utils/Utils.h"
#include "iree/compiler/DispatchCreation/FusionUtils.h"
#include "iree/compiler/DispatchCreation/Passes.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tensor/Transforms/Transforms.h"
#include "mlir/Dialect/Utils/ReshapeOpsUtils.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#define DEBUG_TYPE "iree-dispatch-creation-bubble-up-expand-shapes"

namespace mlir::iree_compiler::DispatchCreation {

#define GEN_PASS_DEF_BUBBLEUPEXPANDSHAPESPASS
#include "iree/compiler/DispatchCreation/Passes.h.inc"

namespace {

/// Consider a simple example:
///
/// %0 = tensor.collapse_shape %arg0 ... AxB -> C
/// %1 = tensor.expand_shape %0      ... C   -> DxE
///
/// The expand can be "bubbled" by first expanding `%arg0` into A0xA1xB and
/// then collapsing back into A0x(A1*B). Let that be MxK, we repeat this process
/// again but the other way to expand out MxK0xK1 and collapse back to
/// (MxK0)*K1. AKA:
///
/// %0 = tensor.expand_shape %arg0   ... AxB       -> (A0xA1)xB
/// %1 = tensor.collapse_shape %0    ... A0x(A1xB) -> MxK
/// %2 = tensor.expand_shape %1      ... MxK       -> Mx(K0xK1)
/// %3 = tensor.collapse_shape %2    ... (MxK0)xK1 -> DxE
///
///

static bool checkReassoc(ArrayRef<ReassociationIndices> reassoc,
                         ArrayRef<int64_t> shape) {
  // TODO: btw shape access is contiguous
  for (ReassociationIndicesRef innerReassoc : reassoc) {
    bool seenDynamic = false;
    for (int64_t dim : innerReassoc) {
      if (ShapedType::isDynamic(shape[dim])) {
        if (seenDynamic) {
          return false;
        }
        seenDynamic = true;
      }
    }
  }
  return true;
}

static bool productOfStaticDims(ArrayRef<int64_t> shape) {
  int64_t product = 1;
  for (int64_t dim : shape) {
    if (dim != ShapedType::kDynamic) {
      product *= dim;
    }
  }
  return product;
}

/// Computes the reassociation needed to expand `inShape` so that it is
/// factorized enough to get collapsed back into `outShape`.
///
/// TODO: let this handle the failure cases.
static SmallVector<ReassociationIndices>
computeExpansion(ArrayRef<int64_t> inShape, ArrayRef<int64_t> outShape) {
  assert(llvm::count(inShape, ShapedType::kDynamic) == 1 &&
         "expected a single dynamic dim");
  assert(llvm::count(outShape, ShapedType::kDynamic) == 1 &&
         "expected a single dynamic dim");

  int64_t inStaticSize = productOfStaticDims(inShape);
  int64_t outStaticSize = productOfStaticDims(outShape);
  (void)inStaticSize;
  (void)outStaticSize;

  int64_t outIdx = 0;
  int64_t numerator = 1;
  int64_t denominator = 1;

  SmallVector<ReassociationIndices> subReassoc;
  for (int64_t inDim : inShape) {
    assert(numerator == 1 || denominator == 1);

    ReassociationIndices currReassoc;
    while (outIdx < outShape.size()) {
      int64_t outDim = outShape[outIdx];
      if (ShapedType::isDynamic(outDim) && ShapedType::isDynamic(inDim)) {
        currReassoc.push_back(outIdx++);
        continue;
      }
      if (ShapedType::isDynamic(outDim)) {
        currReassoc.push_back(outIdx++);
        continue;
      }
      numerator *= outDim;
      currReassoc.push_back(outIdx++);
    }
    subReassoc.push_back(std::move(currReassoc));
  }

  return subReassoc;
}

struct BubbleExpandThroughCollapse
    : public OpRewritePattern<tensor::ExpandShapeOp> {
  using OpRewritePattern<tensor::ExpandShapeOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(tensor::ExpandShapeOp expandOp,
                                PatternRewriter &rewriter) const override {
    auto collapseOp =
        expandOp.getSrc().getDefiningOp<tensor::CollapseShapeOp>();
    if (!collapseOp) {
      return failure();
    }
    SmallVector<ReassociationIndices> collapseReassoc =
        collapseOp.getReassociationIndices();
    SmallVector<ReassociationIndices> expandReassoc =
        expandOp.getReassociationIndices();
    ArrayRef<int64_t> inShape = collapseOp.getSrcType().getShape();
    ArrayRef<int64_t> outShape = expandOp.getType().getShape();
    SmallVector<OpFoldResult> outMixedShape =
        getMixedValues(outShape, expandOp.getOutputShape(), rewriter);

    if (inShape.size() != outShape.size()) {
      llvm::dbgs() << "mismatch input/output rank\n";
      return failure();
    }

    // Only able to handle a single dynamic dimension in the expanded/collapsed
    // reassociation.
    if (!checkReassoc(expandReassoc, outShape) ||
        !checkReassoc(collapseReassoc, inShape)) {
      llvm::dbgs() << "reassoc not invertible\n";
      return failure();
    }

    for (ReassociationIndicesRef reassoc : expandReassoc) {

      SmallVector<int64_t> inGroupShape = llvm::to_vector<6>(
          llvm::map_range(reassoc, [&](int64_t dim) { return inShape[dim]; }));
      SmallVector<int64_t> outGroupShape = llvm::to_vector<6>(
          llvm::map_range(reassoc, [&](int64_t dim) { return outShape[dim]; }));
      auto newReassoc = computeExpansion(inGroupShape, outGroupShape);
      llvm::dbgs() << "reassoc: ";
      for (auto &inner : newReassoc) {
        llvm::dbgs() << " [";
        llvm::interleaveComma(inner, llvm::dbgs());
        llvm::dbgs() << "]";
      }
    }

    llvm::dbgs() << "PASSED!\n";

    return failure();
  }
};

struct BubbleUpExpandShapesPass final
    : public impl::BubbleUpExpandShapesPassBase<BubbleUpExpandShapesPass> {
  void runOnOperation() override;
};

/// Bubbles a `tensor.expand_shape` op through a `tensor.extract_slice` op. This
/// pattern only gets applied when the `extract_slice` doesn't modify dimensions
/// that are expanded by the `expand_shape` and when the `extract_slice` is
/// completely static.
/// TODO: move this upstream with other tensor bubbling patterns.
struct BubbleExpandThroughExtract final
    : public OpRewritePattern<tensor::ExpandShapeOp> {

  using OpRewritePattern<tensor::ExpandShapeOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(tensor::ExpandShapeOp expandOp,
                                PatternRewriter &rewriter) const override {
    auto extractOp = expandOp.getSrc().getDefiningOp<tensor::ExtractSliceOp>();
    if (!extractOp) {
      return failure();
    }

    auto srcType = extractOp.getSourceType();
    auto extractedType = extractOp.getType();
    auto expandedType = expandOp.getType();

    if (srcType.getRank() != extractedType.getRank()) {
      return rewriter.notifyMatchFailure(
          extractOp, "Rank reducing extract_slice not supported");
    }

    if (!srcType.hasStaticShape() || !extractedType.hasStaticShape() ||
        !expandedType.hasStaticShape()) {
      return failure();
    }

    auto reassoc = expandOp.getReassociationIndices();
    for (auto i : llvm::seq<uint64_t>(0, extractedType.getRank())) {
      if (reassoc[i].size() == 1) {
        continue;
      }

      if (srcType.getShape()[i] != extractedType.getShape()[i]) {
        return rewriter.notifyMatchFailure(
            extractOp, "Extract modifies the expanded dimension");
      }
    }

    SmallVector<int64_t> newExpandShape;
    SmallVector<int64_t> offsets;
    SmallVector<int64_t> sizes;
    SmallVector<int64_t> strides;
    for (auto [inDim, outDims] : llvm::enumerate(reassoc)) {
      if (outDims.size() == 1) {
        newExpandShape.push_back(srcType.getShape()[inDim]);
        offsets.push_back(extractOp.getStaticOffsets()[inDim]);
        sizes.push_back(extractOp.getStaticSizes()[inDim]);
        strides.push_back(extractOp.getStaticStrides()[inDim]);
      } else {
        for (auto outDim : outDims) {
          newExpandShape.push_back(expandedType.getShape()[outDim]);
          offsets.push_back(0);
          sizes.push_back(expandedType.getShape()[outDim]);
          strides.push_back(1);
        }
      }
    }

    Type newExpandType =
        RankedTensorType::get(newExpandShape, expandedType.getElementType());
    auto newExpand = rewriter.create<tensor::ExpandShapeOp>(
        expandOp.getLoc(), newExpandType, extractOp.getSource(), reassoc);

    rewriter.replaceOpWithNewOp<tensor::ExtractSliceOp>(
        expandOp, expandedType, newExpand, ValueRange{}, ValueRange{},
        ValueRange{}, offsets, sizes, strides);
    return success();
  }
};

} // namespace

void BubbleUpExpandShapesPass::runOnOperation() {
  MLIRContext *context = &getContext();

  RewritePatternSet bubbleExpandShapePatterns(context);
  linalg::ControlFusionFn bubbleUpExpansionControlFn =
      [](OpOperand *fusedOperand) {
        Operation *producer = fusedOperand->get().getDefiningOp();
        Operation *consumer = fusedOperand->getOwner();
        if (!IREE::Flow::isNonNullAndOutsideDispatch({producer, consumer})) {
          return false;
        }

        // Do not fuse by expand if consumer is dequant.
        if (IREE::LinalgExt::isBitExtendOp(consumer)) {
          return false;
        }

        // Do not fuse producer generic op if it has more than one user
        // or any reduction iterators.
        if (auto producerGenericOp = dyn_cast<linalg::GenericOp>(producer)) {
          return producerGenericOp->hasOneUse() &&
                 llvm::all_of(producerGenericOp.getIteratorTypesArray(),
                              linalg::isParallelIterator);
        }

        // Do not fuse with any producer linalg named ops for now.
        if (isa<linalg::LinalgOp>(producer)) {
          return false;
        }

        // Do not fuse with consumer linalg named ops or reductions.
        if (auto consumerLinalgOp = dyn_cast<linalg::LinalgOp>(consumer)) {
          return isa<linalg::GenericOp>(consumerLinalgOp) &&
                 llvm::all_of(consumerLinalgOp.getIteratorTypesArray(),
                              linalg::isParallelIterator);
        }
        // Fuse in all other cases.
        return true;
      };
  linalg::populateFoldReshapeOpsByExpansionPatterns(bubbleExpandShapePatterns,
                                                    bubbleUpExpansionControlFn);

  // TODO(#19263): Temporary fix to prevent compilation failures when the k2
  // dims get expanded to unit dimensions. This adds the constraint to
  // `bubbleUpExpansionControlFn` that the k2 dimensions cannot be expanded by
  // the reshape fusion.
  linalg::ControlFusionFn linalgExtExpansionFn = [&](OpOperand *fusedOperand) {
    if (!bubbleUpExpansionControlFn(fusedOperand)) {
      return false;
    }

    // There is no need to handle `expand_shape` ops because they would be the
    // producer and therefore are unable to expand the k2 dims.
    auto collapseOp =
        dyn_cast<tensor::CollapseShapeOp>(fusedOperand->get().getDefiningOp());
    auto attentionOp =
        dyn_cast<IREE::LinalgExt::AttentionOp>(fusedOperand->getOwner());
    if (!collapseOp || !attentionOp) {
      return true;
    }

    SmallVector<ReassociationIndices> reassoc =
        collapseOp.getReassociationIndices();
    auto opDetail = IREE::LinalgExt::AttentionOpDetail::get(
        attentionOp.getQueryMap(), attentionOp.getKeyMap(),
        attentionOp.getValueMap(), attentionOp.getOutputMap());

    // Don't sink the `collapse_shape` op if it is collapsing into any of the k2
    // dimensions.
    AffineMap operandMap = attentionOp.getMatchingIndexingMap(fusedOperand);
    for (auto dim : opDetail->getK2Dims()) {
      auto dimExpr = getAffineDimExpr(dim, operandMap.getContext());
      if (std::optional<int64_t> maybeDim =
              operandMap.getResultPosition(dimExpr);
          maybeDim && reassoc[maybeDim.value()].size() > 1) {
        return false;
      }
    }
    return true;
  };
  IREE::LinalgExt::populateFoldReshapeOpsByExpansionPatterns(
      bubbleExpandShapePatterns, linalgExtExpansionFn);

  // Add patterns to do some additional cleanup (on top of canonicalizations
  // that can be done later) of reshape ops.
  tensor::populateFoldTensorEmptyPatterns(bubbleExpandShapePatterns);
  bubbleExpandShapePatterns.insert<BubbleExpandThroughExtract>(context);
  tensor::ExpandShapeOp::getCanonicalizationPatterns(bubbleExpandShapePatterns,
                                                     context);
  bubbleExpandShapePatterns.insert<BubbleExpandThroughCollapse>(context);

  GreedyRewriteConfig rewriteConfig;
  rewriteConfig.maxIterations = GreedyRewriteConfig::kNoLimit;
  if (failed(applyPatternsGreedily(getOperation(),
                                   std::move(bubbleExpandShapePatterns),
                                   rewriteConfig))) {
    getOperation()->emitOpError("Failed to perform elementwise operations");
    return signalPassFailure();
  }
}

} // namespace mlir::iree_compiler::DispatchCreation
