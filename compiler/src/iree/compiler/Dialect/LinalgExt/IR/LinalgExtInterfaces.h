// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_DIALECT_LINALGEXT_IR_LINALGEXTINTERFACES_H_
#define IREE_COMPILER_DIALECT_LINALGEXT_IR_LINALGEXTINTERFACES_H_

#include "llvm/ADT/ArrayRef.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/IR/LinalgInterfaces.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Interfaces/DestinationStyleOpInterface.h"

#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Support/LLVM.h"

namespace mlir::iree_compiler::IREE::LinalgExt {
class LinalgExtOp;

namespace detail {
LogicalResult verifyLinalgExtOpInterface(Operation *op);
}

/// Include the generated interface declarations.
#include "iree/compiler/Dialect/LinalgExt/IR/LinalgExtInterfaces.h.inc" // IWYU pragma: export

/// Returns true if the operation increases/decreases bitwidths of tensors.
/// This function checks that the genericOp:
/// 1. Has only one output.
/// 2. Has all parallel loops.
/// 3. Compared to the element type of the input with highest rank,
///    the output element type has either a higher or lower bitwidth.
inline bool isBitExtendOp(Operation *op) {
  auto bitWidthChangeOp = dyn_cast<BitWidthChangeOpInterface>(op);
  return bitWidthChangeOp && bitWidthChangeOp.isExtensionOp();
}

inline bool isBitTruncateOp(Operation *op) {
  auto bitWidthChangeOp = dyn_cast<BitWidthChangeOpInterface>(op);
  return bitWidthChangeOp && bitWidthChangeOp.isTruncationOp();
}

} // namespace mlir::iree_compiler::IREE::LinalgExt

#include "iree/compiler/Dialect/LinalgExt/IR/LinalgExtOps.h.inc" // IWYU pragma: export

#endif // IREE_COMPILER_DIALECT_LINALGEXT_IR_LINALGEXTINTERFACES_H_
