// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/Flow/Transforms/Passes.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/Linalg/Utils/Utils.h"
#include "mlir/Dialect/Tensor/Transforms/Transforms.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir::iree_compiler::IREE::Flow {

#define GEN_PASS_DEF_CANONICALIZERPASS
#include "iree/compiler/Dialect/Flow/Transforms/Passes.h.inc"

namespace {

static std::optional<SmallVector<OpFoldResult>> getDefiningMixedSizes(Value v) {
  if (auto empty = v.getDefiningOp<tensor::EmptyOp>()) {
    return empty.getMixedSizes();
  } else if (auto extract = v.getDefiningOp<tensor::ExtractSliceOp>()) {
    // TODO: Support rank reducing cases.
    if (extract.getSourceType().getRank() !=
        extract.getResultType().getRank()) {
      return {};
    }
    return extract.getMixedSizes();
  }
  return {};
}

struct FoldFullInsertSlice : public OpRewritePattern<tensor::InsertSliceOp> {
  using OpRewritePattern<tensor::InsertSliceOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(tensor::InsertSliceOp insertSliceOp,
                                PatternRewriter &rewriter) const override {
    if (!insertSliceOp.hasUnitStride() || !insertSliceOp.hasZeroOffset()) {
      return rewriter.notifyMatchFailure(insertSliceOp,
                                         "non-unit stride or non-zero offset.");
    }

    RankedTensorType sourceType = insertSliceOp.getSourceType();
    RankedTensorType resultType = insertSliceOp.getResultType();
    if (sourceType != resultType) {
      return rewriter.notifyMatchFailure(
          insertSliceOp,
          "unimplemented: Cast-like or reshape-like insert ops.");
    }

    std::optional<SmallVector<OpFoldResult>> mixedSizes =
        getDefiningMixedSizes(insertSliceOp.getDest());
    if (!mixedSizes) {
      return rewriter.notifyMatchFailure(
          insertSliceOp, "Could not find producer with list of tensor sizes.");
    }

    for (auto [insertSize, destSize] :
         llvm::zip_equal(insertSliceOp.getMixedSizes(), mixedSizes.value())) {
      if (isa<Value>(insertSize) || isa<Value>(destSize)) {
        if (insertSize != destSize) {
          return rewriter.notifyMatchFailure(insertSliceOp,
                                             "dynamic size mismatch");
        }
        continue;
      }

      // `getMixedSizes` for different ops returns different attribute types
      // (`index` or `i64`) so we compare the values of the ints directly here.
      int64_t staticInsertSize = getConstantIntValue(insertSize).value();
      int64_t staticDestSize = getConstantIntValue(insertSize).value();
      if (staticInsertSize != staticDestSize) {
        return rewriter.notifyMatchFailure(insertSliceOp,
                                           "static size mismatch");
      }
    }

    rewriter.replaceOp(insertSliceOp, insertSliceOp.getSource());
    return success();
  }
};

//===----------------------------------------------------------------------===//
// ElementwiseOpInterchangePattern
//===----------------------------------------------------------------------===//

// If possible, interchange indexing maps to make input maps to remove
// permutations.
struct ElementwiseOpInterchangePattern final
    : public OpRewritePattern<linalg::GenericOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(linalg::GenericOp genericOp,
                                PatternRewriter &rewriter) const override {
    if (!linalg::isElementwise(genericOp) || genericOp.getNumResults() != 1 ||
        genericOp.getNumDpsInputs() == 0)
      return failure();

    // Find an input map that is a non-identity permutation that matches the
    // output map (also not identity).
    AffineMap inputMap;
    auto *initOperand = genericOp.getDpsInitOperand(0);
    AffineMap outputMap = genericOp.getMatchingIndexingMap(initOperand);
    auto inputMaps = genericOp.getIndexingMapsArray();
    for (auto candidateInputMap : ArrayRef(inputMaps).drop_back()) {
      if (!candidateInputMap.isIdentity() &&
          candidateInputMap.isPermutation() && candidateInputMap == outputMap) {
        inputMap = candidateInputMap;
        break;
      }
    }
    if (!inputMap) {
      return failure();
    }

    ArrayRef<AffineExpr> exprs = inputMap.getResults();
    auto perm = llvm::map_to_vector(exprs, [](AffineExpr e) -> unsigned {
      return cast<AffineDimExpr>(e).getPosition();
    });

    // Don't let this mess up other maps.
    for (auto map : genericOp.getIndexingMapsArray()) {
      AffineMap composed = map.compose(inversePermutation(inputMap));
      if (inputMap != map && !compressUnusedDims(composed).isIdentity()) {
        return failure();
      }
    }

    return linalg::interchangeGenericOp(rewriter, genericOp, perm);
  }
};

/// Canonicalize operations in nested regions.
struct CanonicalizerPass
    : public impl::CanonicalizerPassBase<CanonicalizerPass> {
  using IREE::Flow::impl::CanonicalizerPassBase<
      CanonicalizerPass>::CanonicalizerPassBase;
  /// Initialize the canonicalizer by building the set of patterns used during
  /// execution.
  LogicalResult initialize(MLIRContext *context) override {
    // Inherit the same config defaults from the upstream canonicalizer pass.
    config.useTopDownTraversal = true;
    config.enableRegionSimplification = mlir::GreedySimplifyRegionLevel::Normal;

    RewritePatternSet owningPatterns(context);
    for (auto *dialect : context->getLoadedDialects())
      dialect->getCanonicalizationPatterns(owningPatterns);
    for (RegisteredOperationName op : context->getRegisteredOperations())
      op.getCanonicalizationPatterns(owningPatterns, context);

    // Pull in some borderline/downstream canonicalizations for the Flow
    // compilation phase.
    tensor::populateMergeConsecutiveInsertExtractSlicePatterns(owningPatterns);
    owningPatterns.add<FoldFullInsertSlice>(context);
    owningPatterns.add<ElementwiseOpInterchangePattern>(context);

    patterns =
        std::make_shared<FrozenRewritePatternSet>(std::move(owningPatterns));
    return success();
  }
  void runOnOperation() override {
    // Canonicalization is best-effort. Non-convergence is not a pass failure.
    LogicalResult didConverge =
        applyPatternsGreedily(getOperation(), *patterns, config);
    if (this->testConvergence && failed(didConverge)) {
      getOperation()->emitError("Canonicalizer failed to converge");
      return signalPassFailure();
    }
  }
  GreedyRewriteConfig config;
  std::shared_ptr<const FrozenRewritePatternSet> patterns;
};

} // namespace

} // namespace mlir::iree_compiler::IREE::Flow
