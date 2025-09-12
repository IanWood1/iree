// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_GLOBALOPTIMIZATION_PASSES_H_
#define IREE_COMPILER_GLOBALOPTIMIZATION_PASSES_H_

#include <functional>
#include <string>
#include <vector>

#include "llvm/Support/CommandLine.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"

namespace mlir::iree_compiler::GlobalOptimization {

struct TransformOptions : public PassPipelineOptions<TransformOptions> {
  // Maximum byte size increase allowed for constant expr hoisting policy to
  // allow hoisting. The threshold is 1MB by default.
  int64_t constExprMaxSizeIncreaseThreshold = 1024 * 1024;

  // File paths to archives to import parameters from with an optional
  // `scope=` prefix.
  std::vector<std::string> parameterImportPaths;
  // List of parameter keys to import. Any matching keys from any scope will be
  // imported.
  std::vector<std::string> parameterImportKeys;
  // Maximum size of parameters to import or 0 to disable automatic import.
  int64_t parameterImportMaximumSize = 0;

  // File path to an archive to export parameters to with an optional
  // `scope=` prefix.
  std::string parameterExportPath;
  // Minimum size of constants to export as parameters.
  int64_t parameterExportMinimumSize = 0;

  // File path to create a splat parameter archive out of all parameters in the
  // module.
  std::string parameterSplatExportFile = "";

  // Enables aggressive propagation of transposes to the inputs of named ops,
  // rewriting named ops as fused generics.
  bool aggressiveTransposePropagation = false;

  // Enables transposing all concatenations to the outer most dimension.
  bool outerDimConcat = false;

  // Enables data tiling in global optimization phase. There are two data-tiling
  // flags during the transition state. The other has to be off if this one is
  // enabled. Any feature built on top of this path will be deprecated.
  bool dataTiling = true;

  // Enables const-expr hoisting into globals.
  bool constExprHoisting = true;

  // Enables recursive evaluation of immutable globals using the compiler
  // and runtime.
  bool constEval = true;

  // Optimizations to reduce numeric precision where it is safe to do so.
  bool numericPrecisionReduction = false;

  // Strips debug assertions after any useful information has been extracted.
  bool stripAssertions = false;

  // Converts linalg named matmul ops to linalg generic ops.
  bool generalizeMatmul = false;

  // Hook to populate a constant evaluation pass pipeline. If nullptr, then
  // no passes are added for constant evaluation. This must be injected in
  // because constant-evaluators can depend on the whole compiler, of which
  // this is a part, and we maintain strict optionality for this component.
  std::function<void(OpPassManager &passManager)> buildConstEvalPassPipeline;
};

/// Subset of the overall pass pipeline for optimizing globals and numerics.
/// We may ultimately break this out separately so creating a syntactic
/// distinction to keep that as an option.
void buildGlobalOptimizationPassPipeline(
    OpPassManager &mainPassManager, const TransformOptions &transformOptions);

//------------------------------------------------------------------------------
// Wrappers that not use tablegen options.
//------------------------------------------------------------------------------

std::unique_ptr<Pass> createDecomposeConcatPass(bool enableConcatTransposition);

// Used by the demoteContractionInputsToBF16 pass to determine which op inputs
// to demote.
enum class DemotionOption { All, Conv, Matmul, None };
std::unique_ptr<Pass>
createDemoteContractionInputsToBF16Pass(DemotionOption option);

std::unique_ptr<InterfacePass<mlir::FunctionOpInterface>>
createPropagateLinalgTransposePass(bool enableAggressivePropagation);

//----------------------------------------------------------------------------//
// Register GlobalOptimization Passes
//----------------------------------------------------------------------------//

#define GEN_PASS_DECL
#include "iree/compiler/GlobalOptimization/Passes.h.inc" // IWYU pragma: keep

void registerGlobalOptimizationPipeline();

} // namespace mlir::iree_compiler::GlobalOptimization

#endif // IREE_COMPILER_GLOBALOPTIMIZATION_PASSES_H_
