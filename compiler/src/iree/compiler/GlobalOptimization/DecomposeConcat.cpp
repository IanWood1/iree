// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <cstdint>
#include <optional>
#include "iree/compiler/GlobalOptimization/PassDetail.h"
#include "iree/compiler/GlobalOptimization/Passes.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tensor/Transforms/Transforms.h"
#include "mlir/Dialect/Utils/IndexingUtils.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir::iree_compiler::GlobalOptimization {
namespace {

static Value createTranspose(OpBuilder &builder, Value source,
                             SmallVector<int64_t> perm) {
  SmallVector<OpFoldResult> mixedSizes =
      tensor::getMixedSizes(builder, source.getLoc(), source);
  applyPermutationToVector(mixedSizes, perm);
  Type elemType = cast<RankedTensorType>(source.getType()).getElementType();
  Value empty =
      builder.create<tensor::EmptyOp>(source.getLoc(), mixedSizes, elemType)
          .getResult();
  return builder
      .create<linalg::TransposeOp>(source.getLoc(), source, empty, perm)
      ->getResult(0);
}

static int64_t findOuterMostNonUnitDim(ArrayRef<int64_t> &shape) {
  int64_t outerMostNonUnitDim = 0;
  while (outerMostNonUnitDim < shape.size()) {
    if (shape[outerMostNonUnitDim] != 1)
      break;
    outerMostNonUnitDim++;
  }
  return outerMostNonUnitDim;
}

// Transposes the concatenation dimension to happen along the outer most
// non-unit dim of the inputs. The idea is that outer dim concatentations
// can lower to `flow.tensor.update` and ideally disappear, in the worst case
// becoming a sequence of copies. The hope then is that the transposes on the
// inputs and output is then fusable with surrounding operations.
struct TransposeInnerConcatenation : public OpRewritePattern<tensor::ConcatOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(tensor::ConcatOp concatOp,
                                PatternRewriter &rewriter) const override {
    // Get the outer most non-unit dim to transpose to.
    RankedTensorType concatType = concatOp.getResultType();
    ArrayRef<int64_t> concatShape = concatType.getShape();
    int64_t outerMostNonUnitDim = findOuterMostNonUnitDim(concatShape);

    // Nothing to do if the concat is already the outer most non-unit
    int64_t dim = concatOp.getDim();
    if (dim <= outerMostNonUnitDim) {
      return failure();
    }

    SmallVector<int64_t> permutation = computePermutationVector(
        concatOp.getRank(), {dim}, {outerMostNonUnitDim});
    SmallVector<Value> transposedInputs;
    for (auto input : concatOp.getInputs()) {
      transposedInputs.push_back(createTranspose(rewriter, input, permutation));
    }

    SmallVector<int64_t> newShape = applyPermutation(concatShape, permutation);
    auto newConcatType = RankedTensorType::get(
        newShape, concatOp.getResultType().getElementType());
    Value newConcat = rewriter.create<tensor::ConcatOp>(
        concatOp.getLoc(), newConcatType, /*dim=*/outerMostNonUnitDim,
        transposedInputs);
    auto invPerm = invertPermutationVector(permutation);
    Value transposedConcat = createTranspose(rewriter, newConcat, invPerm);
    rewriter.replaceOp(concatOp, transposedConcat);
    return success();
  }
};

// Same as `TransposeInnerConcatenation` except operating on
// `tensor.extract_slice` ops instead of `tensor.concat`. Note that unlike
// `ConcatOp`, the rank of the result can be less than the source tensor's rank
struct TransposeInnerExtractSlice
    : public OpRewritePattern<tensor::ExtractSliceOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(tensor::ExtractSliceOp sliceOp,
                                PatternRewriter &rewriter) const override {

    RankedTensorType sourceType = sliceOp.getSourceType();
    ArrayRef<int64_t> sourceShape = sourceType.getShape();
    ArrayRef<int64_t> extractShape = sliceOp.getResultType().getShape();
    if (llvm::is_contained(extractShape, ShapedType::kDynamic) ||
        llvm::is_contained(sourceShape, ShapedType::kDynamic)) {
      return failure();
    }

    SmallVector<int64_t> sizes(sliceOp.getStaticSizes());

    int64_t outerMostNonUnitDim = findOuterMostNonUnitDim(sourceShape);
    int64_t firstShrunkDim = 0;
    while (firstShrunkDim < sizes.size()) {
      if (sizes[firstShrunkDim] != sourceShape[firstShrunkDim])
        break;
      firstShrunkDim++;
    }

    // Match agains `extract_slice` that extracts the full size from the first
    // N-1 dims. Also, since source & result aren't dynamic, neither is an
    // element in [0, N-1] of `sizes`
    if (firstShrunkDim != sourceShape.size() - 1 ||
        outerMostNonUnitDim == firstShrunkDim)
      return failure();

    SmallVector<int64_t> permutation = computePermutationVector(
        sourceShape.size(), {firstShrunkDim}, {outerMostNonUnitDim});
    SmallVector<int64_t> f32 = applyPermutation(sourceShape, permutation);

    Value transposedSource =
        createTranspose(rewriter, sliceOp.getSource(), permutation);

    auto maybeMask = sliceOp.computeRankReductionMask();
    llvm::SmallDenseSet<unsigned> rankReducingMask;
    if (maybeMask)
      rankReducingMask = *maybeMask;

    // Find rank reducing map in the pre-transposed domain.
    int64_t dim = 0;
    llvm::SmallDenseMap<int64_t, int64_t> rankReducedMap;
    // Since `dim` is in the pre-transposed domain, and is incrementing each
    // iteration, `idx` must also be in the pre-transposed domain.
    for (int64_t idx = 0, e = extractShape.size(); idx < e; ++idx) {
      // Get index in the transposed domain, since `rankReducingMask` is in
      // the transposed domain.
      if (!rankReducingMask.contains(permutation[idx])) {
        // Domain of `rankReducedMap` is in pre-transposed domain.
        rankReducedMap[idx] = dim++;
      }
    }

    // Compute the new permutation by dropping all rank-reduced dimensions.
    SmallVector<int64_t> rankReducedPerm;
    for (int64_t i : permutation) {
      if (!rankReducingMask.contains(i)) {
        rankReducedPerm.push_back(rankReducedMap[i]);
      }
    }
    assert(extractShape.size() == rankReducedPerm.size());
    auto rankReducedTransShape =
        applyPermutation(extractShape, rankReducedPerm);

    auto newSliceOp = rewriter.create<tensor::ExtractSliceOp>(
        sliceOp.getLoc(),
        RankedTensorType::get(rankReducedTransShape,
                              sliceOp.getResultType().getElementType()),
        transposedSource,
        applyPermutation(sliceOp.getMixedOffsets(), permutation),
        applyPermutation(sliceOp.getMixedSizes(), permutation),
        sliceOp.getMixedStrides());

    if (rankReducingMask.contains(firstShrunkDim)) {
      rewriter.replaceOp(sliceOp, newSliceOp);
      return success();
    }

    auto rankReducedInvPerm = invertPermutationVector(rankReducedPerm);
    Value transposedExtract =
        createTranspose(rewriter, newSliceOp.getResult(), rankReducedInvPerm);
    rewriter.replaceOp(sliceOp, transposedExtract);
    return success();
  }
};

struct DecomposeConcatPass : public DecomposeConcatBase<DecomposeConcatPass> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<linalg::LinalgDialect>();
  }
  DecomposeConcatPass(bool enableConcatTransposition) {
    this->enableConcatTransposition = enableConcatTransposition;
  }
  DecomposeConcatPass(const DecomposeConcatPass &pass)
      : DecomposeConcatPass(pass.enableConcatTransposition) {}

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    RewritePatternSet patterns(context);

    if (enableConcatTransposition) {
      patterns.insert<TransposeInnerConcatenation, TransposeInnerExtractSlice>(
          context, /*benefit=*/2);
    }
    tensor::populateDecomposeTensorConcatPatterns(patterns);
    if (failed(applyPatternsAndFoldGreedily(getOperation(),
                                            std::move(patterns)))) {
      return signalPassFailure();
    }
  }
};

} // namespace

std::unique_ptr<Pass>
createDecomposeConcatPass(bool enableConcatTransposition) {
  return std::make_unique<DecomposeConcatPass>(enableConcatTransposition);
}

} // namespace mlir::iree_compiler::GlobalOptimization
