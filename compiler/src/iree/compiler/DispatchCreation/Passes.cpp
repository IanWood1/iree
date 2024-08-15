// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/DispatchCreation/Passes.h"

#include "iree/compiler/Codegen/Common/PassUtils.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Pass/PassOptions.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Transforms/Passes.h"

//===----------------------------------------------------------------------===//
// Command Line Options
//===----------------------------------------------------------------------===//

static llvm::cl::opt<std::string> clDispatchTransformFileName(
    "iree-flow-dispatch-use-transform-dialect",
    llvm::cl::desc("MLIR file containing a top-level module that specifies "
                   "the transformations to apply to form dispatch regions."),
    llvm::cl::init(""));

static llvm::cl::opt<bool> clDetensoring(
    "iree-flow-enable-detensoring",
    llvm::cl::desc(
        "Enable changing of tensor operations into scalar operations."),
    llvm::cl::init(false));

static llvm::cl::opt<bool> clEnableElementWiseFuseMultiReduction(
    "iree-flow-element-wise-fuse-multi-reduction",
    llvm::cl::desc("Enable element-wise fusion of multi-reduction loop ops."),
    llvm::cl::init(true));

static llvm::cl::opt<bool> clEnableFusePaddingIntoLinalgConsumerOps(
    "iree-flow-enable-fuse-padding-into-linalg-consumer-ops",
    llvm::cl::desc("Enable fusing tensor.pad ops into Linalg consumer ops."),
    llvm::cl::init(false));

static llvm::cl::opt<bool> clEnableFusePaddingIntoLinalgProducerOps(
    "iree-flow-enable-fuse-padding-into-linalg-producer-ops",
    llvm::cl::desc("Enable fusing tensor.pad ops into Linalg consumer ops."),
    llvm::cl::init(false));

static llvm::cl::opt<bool> clEnableFuseHorizontalContractions(
    "iree-flow-enable-fuse-horizontal-contractions",
    llvm::cl::desc(
        "Enables horizontal fusion of contractions with one common operand"),
    llvm::cl::init(false));

static llvm::cl::opt<bool> clCollapseReductionDims(
    "iree-flow-collapse-reduction-dims",
    llvm::cl::desc("Enable collapsing of reduction dims"),
    llvm::cl::init(false));

static llvm::cl::opt<bool>
    clEnableFuseMultiUse("iree-flow-fuse-multi-use",
                         llvm::cl::desc("Fuse multi-use ops."),
                         llvm::cl::init(false));

static llvm::cl::opt<bool> clEnableAggressiveFusion(
    "iree-flow-enable-aggressive-fusion",
    llvm::cl::desc("Aggressive fusion opportunities that are behind a flag "
                   "since all backends dont support it yet"),
    llvm::cl::init(false));

static llvm::cl::opt<bool> clEnableDataTiling(
    "iree-flow-experimental-data-tiling",
    llvm::cl::desc("Enable data-tiling at flow level, i.e., it sets encodings "
                   "in dispatch regions, hoist them out of region, and enables "
                   "fusion for the set_encodings. This is still an "
                   "experimental path. The current main data tiling path is "
                   "iree-opt-data-tiling, which is on by default. To use this "
                   "path, --iree-opt-data-tiling=false must be set as wells"),
    llvm::cl::init(false));

namespace mlir::iree_compiler::DispatchCreation {

static std::unique_ptr<Pass>
createCanonicalizerPass(const GreedyRewriteConfig &config,
                        ArrayRef<std::string> disabledPatterns,
                        ArrayRef<std::string> enabledPatterns) {
  assert(false && "TODO");
}

//===----------------------------------------------------------------------===//
// Pipelines
//===----------------------------------------------------------------------===//

void addDispatchRegionCreationPreprocessingPasses(OpPassManager &passManager) {
  // 1. Do some simple elementwise op fusion. This could be skipped,
  //    but could reduce the surface area of ops to handle later.
  FunctionLikeNest(passManager)
      .addPass([]() {
        return DispatchCreation::createElementwiseOpFusionPass(
            ElementwiseOpFusionPassOptions{
                clEnableElementWiseFuseMultiReduction});
      })
      .addPass(DispatchCreation::createCanonicalizerPass)
      .addPass(mlir::createCSEPass)

      // 2. Bubble up expand_shape ops (or sink collapse_shape ops) to get
      //    elementwise operation into higher dimensions for more fusion
      //    opportunities.
      .addPass(DispatchCreation::createBubbleUpExpandShapesPass)
      .addPass(DispatchCreation::createCanonicalizerPass)
      .addPass(mlir::createCSEPass)

      // 3. Perform elementwise operation fusion again (now with higher
      //    dimensionality).
      .addPass([]() {
        return DispatchCreation::createElementwiseOpFusionPass(
            ElementwiseOpFusionPassOptions{
                clEnableElementWiseFuseMultiReduction});
      })
      .addPass(DispatchCreation::createCanonicalizerPass)
      .addPass(mlir::createCSEPass)

      // 4. After elementwise operation fusion sink reshapes that block
      //    producer-consumer fusion.
      .addPass(DispatchCreation::createSinkReshapesPass)
      .addPass(DispatchCreation::createCanonicalizerPass)
      .addPass(mlir::createCSEPass);

  if (clEnableFuseHorizontalContractions) {
    FunctionLikeNest(passManager)
        .addPass(createFuseHorizontalContractionsPass)
        .addPass(mlir::createCanonicalizerPass)
        .addPass(mlir::createCSEPass);
  }

  FunctionLikeNest(passManager)
      // 5. After all the reshape propagations, fuse elementwise operations
      //    even if the producer has multiple uses.
      .addPass(DispatchCreation::createFuseMultiUseElementwiseProducerPass)

      // 6. Some more "post elementwise fusion passes".
      //    a. Detensorize.
      //       TODO: This is probably not in the right place.
      .addPredicatedPass(clDetensoring,
                         [&]() { return mlir::createLinalgDetensorizePass(); })
      .addPass(DispatchCreation::createCanonicalizerPass)
      .addPass(mlir::createCSEPass)

      //    b. For ops with multiple reduction dimensions, collapse the
      //       reduction dimension.
      //       TODO: This pass is only needed till all backends can handle
      //       multiple reduction dimensions.
      .addPredicatedPass(
          clCollapseReductionDims,
          DispatchCreation::createCollapseReductionDimensionsPass)

      //     c. Split reduction operations into parallel and reduction, i.e
      //        .
      .addPass(DispatchCreation::createSplitReductionPass)

      //     d. Transpose generic ops to
      //        - help with dispatch region formation.
      //        - move reduction iterators to be innermost.
      .addPass(DispatchCreation::createTransposeGenericOpsPass);
}

// Pipeline to first create `flow.dispatch.region` ops and then lower to
// `flow.dispatch.workgroup` ops.
static void addDispatchRegionCreationPasses(OpPassManager &passManager) {
  FunctionLikeNest(passManager)
      // Only want use the transform dialect for some dispatch regions and let
      // the FormDispatchRegions handle the rest. This only moves the root
      // compute op into the dispatch region, so that we can run additional
      // transformations afterwards with a simple region and without bothering
      // producers.
      .addPredicatedPass(
          !clDispatchTransformFileName.empty(),
          [&]() {
            DispatchWithTransformDialectPassOptions options;
            options.transformSpecPath = clDispatchTransformFileName;
            return createDispatchWithTransformDialectPass(options);
          })
      // Create dispatches for scalar operations as roots
      .addPass(DispatchCreation::createFormScalarDispatchesPass)
      // Create `flow.dispatch.region` centered around a root and fuse with
      // producers and consumers.
      .addPass([&]() {
        return DispatchCreation::createFormDispatchRegionsPass(
            FormDispatchRegionsPassOptions{
                clEnableAggressiveFusion,
                clEnableFusePaddingIntoLinalgConsumerOps,
                clEnableFusePaddingIntoLinalgProducerOps});
      })
      // Clone all producers into the dispatch region to perpare for being
      // isolated from above. This enables running additional transformations
      // afterwards that would need the full dispatch content but don't want to
      // handle explicit captures as materialized as dispatch workgroup operands
      // and block arguments.
      .addPass(DispatchCreation::createCloneProducersIntoDispatchRegionsPass)
      .addPredicatedPass(clEnableDataTiling,
                         [&]() {
                           return createSetEncodingPass(
                               SetEncodingPassOptions{clPadFactor});
                         })
      // Collapse dimensions of linalg Ops.
      .addPass(DispatchCreation::createCollapseDimensionsPass);
}

// Apply preprocessing and form dispatch regions
void buildDispatchCreationPassPipeline(
    OpPassManager &passManager, const TransformOptions &transformOptions) {
  FunctionLikeNest(passManager)
      // Preprocess the input to a form more amenable for fusion.
      .addPass(DispatchCreation::createFusionPreprocessingPass)
      .addPass(DispatchCreation::createCanonicalizerPass)
      .addPass(mlir::createCSEPass);

  addDispatchRegionCreationPreprocessingPasses(passManager);
  addDispatchRegionCreationPasses(passManager);
}

namespace {
#define GEN_PASS_REGISTRATION
#include "iree/compiler/DispatchCreation/Passes.h.inc" // IWYU pragma: keep
} // namespace

void registerDispatchCreationPasses() {
  // Generated from Passes.td
  registerPasses();
}

} // namespace mlir::iree_compiler::DispatchCreation
