// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//===--- FusionUtils.h --- Implementation of fusion utility functions -----===//
//===----------------------------------------------------------------------===//

#include "compiler/src/iree/compiler/Dialect/Flow/Transforms/FusionUtils.h"
#include "compiler/src/iree/compiler/Dialect/Flow/Transforms/RegionOpUtils.h"
#include "llvm/ADT/STLExtras.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"

namespace mlir::iree_compiler::IREE::Flow {

bool areFusableAsElementwiseOps(MLIRContext *context, OpOperand *fusedOperand,
                                bool fuseMultiReduction) {
  Operation *producerOp = fusedOperand->get().getDefiningOp();
  Operation *consumerOp = fusedOperand->getOwner();
  if (!producerOp)
    return false;

  // Check for i1 return types, if so aggressively fuse to avoid `i1` buffers.
  if (llvm::all_of(producerOp->getResultTypes(), [](Type t) {
        if (t.isInteger(1))
          return true;
        if (auto shapedType = llvm::dyn_cast<ShapedType>(t)) {
          if (shapedType.getElementType().isInteger(1))
            return true;
        }
        return false;
      })) {
    return true;
  }

  // Don't fuse if all of the consumer maps aren't projected permutations.
  if (auto linalgConsumerOp = dyn_cast<linalg::LinalgOp>(consumerOp)) {
    if (!llvm::all_of(
            linalgConsumerOp.getIndexingMapsArray(),
            [](AffineMap map) { return map.isProjectedPermutation(); })) {
      return false;
    }
  }

  // If the generic op is "just" copy, then fuse always.
  Block &body = producerOp->getRegion(0).front();
  if (std::begin(body)->hasTrait<OpTrait::IsTerminator>())
    return true;
  if (llvm::all_of(body.getArguments(),
                   [](BlockArgument arg) { return arg.use_empty(); })) {
    // The operands aren't used, its just an `linalg.index` op.
    return true;
  }

  // If producer does not have a single user, dont fuse.
  if (!producerOp->hasOneUse())
    return false;

  // Do no fuse dequantization-like operations with producers. The
  // dequantization ops are cloned into all their use dispatches. So fusing
  // producer with consumer here would then result in producer also getting
  // cloned into many dispatches which is against the thumb rule of fusion to
  // not introduce additional computation (except for dequant ops). If the
  // consumer has only one use, then this fusion is fine since cloning wont
  // result in redundant computation of the producer. (Also note that the
  // producer is always an elementwise operation).
  if (isDequantizationLikeOp(consumerOp) && !consumerOp->hasOneUse()) {
    return false;
  }

  // If the producer has a single use (this op), only fuse if
  // - 1) The consumer op is all parallel loops. The parallelism of the consumer
  //      can be used as a way to amortize cost of redundant computation
  // - 2) If consumer op is a reduction, only fuse if the indexing map in the
  //      consumer for the producer result is a permutation. If it is a
  //      broadcast this ends up redundantly computing operations without more
  //      parallelism.
  if (auto linalgConsumerOp = dyn_cast<linalg::LinalgOp>(consumerOp)) {

    if (linalgConsumerOp.getNumParallelLoops() ==
        linalgConsumerOp.getNumLoops()) {
      return true;
    }
    if (!linalgConsumerOp.getMatchingIndexingMap(fusedOperand)
             .isPermutation()) {
      return false;
    }
    if (!fuseMultiReduction && linalgConsumerOp.getNumReductionLoops() != 1) {
      return false;
    }
    if (linalg::isaContractionOpInterface(linalgConsumerOp) ||
        linalg::isaConvolutionOpInterface(linalgConsumerOp)) {
      return false;
    }
    return true;
  }

  // All other cases dont fuse.
  return false;
}

LogicalResult ExpansionInfo::compute(
    LinalgExt::LinalgFusionOpInterface fusibleOp, OpOperand *fusableOpOperand,
    ArrayRef<AffineMap> reassociationMaps, ArrayRef<int64_t> expandedShape,
    ArrayRef<int64_t> collapsedShape, PatternRewriter &rewriter) {
  if (reassociationMaps.empty())
    return failure();
  AffineMap fusedIndexMap = fusibleOp.getMatchingIndexingMap(fusableOpOperand);

  SmallVector<int64_t, 4> originalLoopRange = fusibleOp.getStaticLoopRanges();
  originalLoopExtent.assign(originalLoopRange.begin(), originalLoopRange.end());

  reassociation.clear();
  expandedShapeMap.clear();
  // Compute the number of dimension in the expanded op that correspond to each
  // dimension of the original op.
  SmallVector<unsigned> numExpandedDims(fusedIndexMap.getNumDims(), 1);
  expandedShapeMap.resize(fusedIndexMap.getNumDims());
  for (const auto &resultExpr : llvm::enumerate(fusedIndexMap.getResults())) {
    unsigned pos = cast<AffineDimExpr>(resultExpr.value()).getPosition();
    AffineMap foldedDims = reassociationMaps[resultExpr.index()];
    numExpandedDims[pos] = foldedDims.getNumResults();
    ArrayRef<int64_t> shape =
        expandedShape.slice(foldedDims.getDimPosition(0), numExpandedDims[pos]);
    expandedShapeMap[pos].assign(shape.begin(), shape.end());
  }
  // The remaining dimensions remain the same.
  for (unsigned i : llvm::seq<unsigned>(0, fusedIndexMap.getNumDims()))
    if (expandedShapeMap[i].empty())
      expandedShapeMap[i] = {originalLoopExtent[i]};

  // Compute reassociation map from the original op to the expanded op.
  unsigned sum = 0;
  reassociation.reserve(fusedIndexMap.getNumDims());
  for (const auto &numFoldedDim : llvm::enumerate(numExpandedDims)) {
    auto seq = llvm::seq<int64_t>(sum, sum + numFoldedDim.value());
    reassociation.emplace_back(seq.begin(), seq.end());
    sum += numFoldedDim.value();
  }
  expandedOpNumDims = sum;
  return success();
}

unsigned ExpansionInfo::getOrigOpNumDims() const {
  return reassociation.size();
}
unsigned ExpansionInfo::getExpandedOpNumDims() const {
  return expandedOpNumDims;
}
ReassociationIndicesRef ExpansionInfo::getExpandedDims(unsigned i) const {
  return reassociation[i];
}
ArrayRef<int64_t> ExpansionInfo::getExpandedShapeOfDim(unsigned i) const {
  return expandedShapeMap[i];
}
ArrayRef<int64_t> ExpansionInfo::getOriginalShape() const {
  return originalLoopExtent;
}

LogicalResult isOpExpandable(LinalgExt::LinalgFusionOpInterface fusibleOp,
                             const ExpansionInfo &expansionInfo,
                             PatternRewriter &rewriter) {
  linalg::LinalgOp maybeLinalgOp =
      dyn_cast<linalg::LinalgOp>(fusibleOp.getOperation());
  if (maybeLinalgOp && !maybeLinalgOp.hasIndexSemantics())
    return success();
  for (unsigned i : llvm::seq<unsigned>(0, expansionInfo.getOrigOpNumDims())) {
    ArrayRef<int64_t> expandedShape = expansionInfo.getExpandedShapeOfDim(i);
    if (expandedShape.size() == 1)
      continue;
    for (int64_t shape : expandedShape.drop_front()) {
      if (ShapedType::isDynamic(shape)) {
        return rewriter.notifyMatchFailure(
            fusibleOp, "cannot expand due to index semantics and dynamic dims");
      }
    }
  }
  return success();
}

AffineMap getIndexingMapInExpandedOp(OpBuilder &builder, AffineMap indexingMap,
                                     const ExpansionInfo &expansionInfo) {
  SmallVector<AffineExpr> newExprs;
  for (AffineExpr expr : indexingMap.getResults()) {
    unsigned pos = cast<AffineDimExpr>(expr).getPosition();
    SmallVector<AffineExpr, 4> expandedExprs = llvm::to_vector<4>(
        llvm::map_range(expansionInfo.getExpandedDims(pos), [&](int64_t v) {
          return builder.getAffineDimExpr(static_cast<unsigned>(v));
        }));
    newExprs.append(expandedExprs.begin(), expandedExprs.end());
  }
  return AffineMap::get(expansionInfo.getExpandedOpNumDims(),
                        indexingMap.getNumSymbols(), newExprs,
                        builder.getContext());
}

RankedTensorType getExpandedType(RankedTensorType originalType,
                                 AffineMap indexingMap,
                                 const ExpansionInfo &expansionInfo) {
  SmallVector<int64_t> expandedShape;
  for (AffineExpr expr : indexingMap.getResults()) {
    unsigned dim = cast<AffineDimExpr>(expr).getPosition();
    auto dimExpansion = expansionInfo.getExpandedShapeOfDim(dim);
    expandedShape.append(dimExpansion.begin(), dimExpansion.end());
  }
  return RankedTensorType::get(expandedShape, originalType.getElementType());
}

SmallVector<ReassociationIndices>
getReassociationForExpansion(AffineMap indexingMap,
                             const ExpansionInfo &expansionInfo) {
  SmallVector<ReassociationIndices> reassociation;
  unsigned numReshapeDims = 0;
  for (AffineExpr expr : indexingMap.getResults()) {
    unsigned dim = cast<AffineDimExpr>(expr).getPosition();
    auto numExpandedDims = expansionInfo.getExpandedDims(dim).size();
    SmallVector<int64_t, 2> indices = llvm::to_vector<2>(
        llvm::seq<int64_t>(numReshapeDims, numReshapeDims + numExpandedDims));
    reassociation.emplace_back(std::move(indices));
    numReshapeDims += numExpandedDims;
  }
  return reassociation;
}

/// Checks if a single dynamic dimension expanded into multiple dynamic
/// dimensions.
LogicalResult
validateDynamicDimExpansion(LinalgExt::LinalgFusionOpInterface interfaceOp,
                            const ExpansionInfo &expansionInfo,
                            PatternRewriter &rewriter) {
  for (unsigned i : llvm::seq<unsigned>(0, expansionInfo.getOrigOpNumDims())) {
    ArrayRef<int64_t> expandedShape = expansionInfo.getExpandedShapeOfDim(i);
    if (expandedShape.size() == 1)
      continue;
    bool foundDynamic = false;
    for (int64_t shape : expandedShape) {
      if (!ShapedType::isDynamic(shape))
        continue;
      if (foundDynamic) {
        return rewriter.notifyMatchFailure(
            interfaceOp, "cannot infer expanded shape with multiple dynamic "
                         "dims in the same reassociation group");
      }
      foundDynamic = true;
    }
  }
  return success();
}

} // namespace mlir::iree_compiler::IREE::Flow
