// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//===- GeneralizeLinalgOps.cpp - Pass to generalize named LinalgOps -------==//
//
// The pass is to generalize Linalg named operations that are better off being
// represented as `linalg.generic` operations in IREE.
//
//===----------------------------------------------------------------------===//

#include "iree/compiler/Dialect/Flow/Transforms/RegionOpUtils.h"
#include "iree/compiler/GlobalOptimization/PassDetail.h"
#include "iree/compiler/GlobalOptimization/Passes.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/IR/LinalgInterfaces.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir::iree_compiler::GlobalOptimization {

namespace {

class GeneralizeBroadcastMatmul
    : public OpInterfaceRewritePattern<linalg::ContractionOpInterface> {
public:
  using OpInterfaceRewritePattern::OpInterfaceRewritePattern;
  LogicalResult matchAndRewrite(linalg::ContractionOpInterface matmulOp,
                                PatternRewriter &rewriter) const override {
    auto linalgOp = dyn_cast<linalg::LinalgOp>(matmulOp.getOperation());
    if (!linalgOp)
      return failure();

    // Find broadcast producer
    unsigned operandNumber;
    linalg::BroadcastOp broadcastOp(nullptr);
    for (auto &currOperand : matmulOp->getOpOperands()) {
      if ((broadcastOp =
               currOperand.get().getDefiningOp<linalg::BroadcastOp>())) {
        operandNumber = currOperand.getOperandNumber();
        break;
      }
    }
    if (!broadcastOp) {
      return rewriter.notifyMatchFailure(matmulOp,
                                         "no broadcast operand found");
    }
    FailureOr<linalg::GenericOp> maybeGeneric =
        linalg::generalizeNamedOp(rewriter, linalgOp);
    if (failed(maybeGeneric)) {
      return rewriter.notifyMatchFailure(matmulOp, "failed to generalize");
    }
    linalg::GenericOp genericOp = *maybeGeneric;

    SmallVector<AffineMap> genericIndexingMaps =
        genericOp.getIndexingMapsArray();
    AffineMap broadcastMap = broadcastOp.getIndexingMapsArray()[0];
    genericIndexingMaps[operandNumber] =
        broadcastMap.compose(genericIndexingMaps[operandNumber]);

    rewriter.startOpModification(genericOp);
    genericOp.getInputsMutable()[operandNumber].set(broadcastOp.getInput());
    genericOp.setIndexingMapsAttr(
        rewriter.getAffineMapArrayAttr(genericIndexingMaps));
    rewriter.finalizeOpModification(genericOp);
    return success();
  }
};

struct GeneralizeLinalgNamedOpsPass
    : public GeneralizeLinalgNamedOpsBase<GeneralizeLinalgNamedOpsPass> {

  void runOnOperation() override;
};
} // namespace

void GeneralizeLinalgNamedOpsPass::runOnOperation() {
  auto funcOp = getOperation();
  SmallVector<linalg::LinalgOp> namedOpCandidates;

  RewritePatternSet patterns(&getContext());
  patterns.insert<GeneralizeBroadcastMatmul>(&getContext());
  if (failed(applyPatternsAndFoldGreedily(funcOp, std::move(patterns)))) {
    return signalPassFailure();
  }

  funcOp.walk([&](linalg::LinalgOp linalgOp) {
    if (!IREE::Flow::isNonNullAndOutsideDispatch(linalgOp)) {
      return;
    }
    if (isa_and_nonnull<linalg::AbsOp, linalg::AddOp, linalg::BroadcastOp,
                        linalg::CeilOp, linalg::CopyOp, linalg::DivOp,
                        linalg::DivUnsignedOp, linalg::ElemwiseBinaryOp,
                        linalg::ElemwiseUnaryOp, linalg::ExpOp, linalg::FloorOp,
                        linalg::LogOp, linalg::MapOp, linalg::MaxOp,
                        linalg::MulOp, linalg::NegFOp, linalg::ReduceOp,
                        linalg::SubOp, linalg::TransposeOp>(
            linalgOp.getOperation())) {
      namedOpCandidates.push_back(linalgOp);
    }
  });

  IRRewriter rewriter(&getContext());
  for (auto linalgOp : namedOpCandidates) {
    rewriter.setInsertionPoint(linalgOp);
    FailureOr<linalg::GenericOp> generalizedOp =
        linalg::generalizeNamedOp(rewriter, linalgOp);
    if (failed(generalizedOp)) {
      linalgOp->emitOpError("failed to generalize operation");
      return signalPassFailure();
    }
  }
}

std::unique_ptr<InterfacePass<mlir::FunctionOpInterface>>
createGeneralizeLinalgNamedOpsPass() {
  return std::make_unique<GeneralizeLinalgNamedOpsPass>();
}

} // namespace mlir::iree_compiler::GlobalOptimization
