// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/Flow/Transforms/Passes.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/Utils.h"
#include "mlir/Dialect/MemRef/Transforms/Transforms.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tensor/Transforms/Transforms.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir::iree_compiler::IREE::Flow {

#define GEN_PASS_DEF_CANONICALIZEPASS
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
  using OpRewritePattern::OpRewritePattern;

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

class DecomposeTensorReshape : public OpRewritePattern<tensor::ReshapeOp> {
public:
  using OpRewritePattern<tensor::ReshapeOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(tensor::ReshapeOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Value input = op.getSource();
    RankedTensorType inputType = cast<RankedTensorType>(input.getType());
    RankedTensorType outputType = cast<RankedTensorType>(op.getResultType());

    // Create a 1D tensor type for the collapsed shape. If the input rank is
    // fully static, make the collapsed extent static; otherwise dynamic.
    int64_t totalElements = 1;
    bool allStatic = true;
    for (int64_t dim : inputType.getShape()) {
      if (ShapedType::isDynamic(dim)) {
        allStatic = false;
        break;
      }
      totalElements *= dim;
    }
    SmallVector<int64_t> collapsedShape = {allStatic ? totalElements
                                                     : ShapedType::kDynamic};
    RankedTensorType collapsedType =
        RankedTensorType::get(collapsedShape, inputType.getElementType());

    // Create reassociation indices to collapse all dimensions into one
    SmallVector<ReassociationIndices> collapseReassociation;
    ReassociationIndices allDims;
    for (int i = 0; i < inputType.getRank(); ++i) {
      allDims.push_back(i);
    }
    collapseReassociation.push_back(allDims);

    // Create the collapse_shape operation
    Value collapsed = tensor::CollapseShapeOp::create(
        rewriter, loc, collapsedType, input, collapseReassociation);

    // Create reassociation indices to expand back to output shape
    SmallVector<ReassociationIndices> expandReassociation;
    ReassociationIndices expandedGroup;
    for (int i = 0; i < outputType.getRank(); ++i) {
      expandedGroup.push_back(i);
    }
    expandReassociation.push_back(expandedGroup);

    // Create the expand_shape operation
    Value expanded = tensor::ExpandShapeOp::create(
        rewriter, loc, outputType, collapsed, expandReassociation);

    // Replace the original reshape operation
    rewriter.replaceOp(op, expanded);
    return success();
  }
};

/// Convert an "affine.apply" operation into a sequence of arith ops.
class AffineApplyLowering : public OpRewritePattern<affine::AffineApplyOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(affine::AffineApplyOp op,
                                PatternRewriter &rewriter) const override {
    auto maybeExpandedMap =
        affine::expandAffineMap(rewriter, op.getLoc(), op.getAffineMap(),
                                llvm::to_vector<8>(op.getOperands()));
    if (!maybeExpandedMap)
      return failure();
    rewriter.replaceOp(op, *maybeExpandedMap);
    return success();
  }
};

/// Canonicalize operations in nested regions.
struct CanonicalizePass : public impl::CanonicalizePassBase<CanonicalizePass> {
  using IREE::Flow::impl::CanonicalizePassBase<
      CanonicalizePass>::CanonicalizePassBase;
  /// Initialize the canonicalizer by building the set of patterns used during
  /// execution.
  LogicalResult initialize(MLIRContext *context) override {
    // Inherit the same config defaults from the upstream canonicalizer pass.
    config.setUseTopDownTraversal().setRegionSimplificationLevel(
        mlir::GreedySimplifyRegionLevel::Normal);

    RewritePatternSet owningPatterns(context);
    for (auto *dialect : context->getLoadedDialects())
      dialect->getCanonicalizationPatterns(owningPatterns);
    for (RegisteredOperationName op : context->getRegisteredOperations())
      op.getCanonicalizationPatterns(owningPatterns, context);

    // Pull in some borderline/downstream canonicalizations for the Flow
    // compilation phase.
    tensor::populateMergeConsecutiveInsertExtractSlicePatterns(owningPatterns);
    owningPatterns.add<FoldFullInsertSlice>(context);
    owningPatterns.add<AffineApplyLowering>(context);
    owningPatterns.add<DecomposeTensorReshape>(context);

    patterns =
        std::make_shared<FrozenRewritePatternSet>(std::move(owningPatterns));
    return success();
  }
  void runOnOperation() override {
    // Canonicalization is best-effort. Non-convergence is not a pass failure.
    config.enableConstantCSE(cseConstants);
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
