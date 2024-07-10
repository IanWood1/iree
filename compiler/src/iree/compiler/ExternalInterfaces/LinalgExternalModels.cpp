// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/ExternalInterfaces/LinalgExternalModels.h"
#include "iree/compiler/Dialect/LinalgExt/IR/LinalgExtInterfaces.h"

#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Interfaces/ValueBoundsOpInterface.h"

namespace mlir::iree_compiler {
namespace {

enum class BitWidthChangeInfo {
  kNull,
  kExtend,
  kTruncate,
};

/// Determines if the operation increases/decreases bitwidths of tensors.
/// This function checks that the genericOp:
/// 1. Has only one output.
/// 2. Has all parallel loops.
/// 3. Compared to the element type of the input with highest rank,
///    the output element type has either a higher or lower bitwidth.
static BitWidthChangeInfo isBitExtendOrTruncateOp(linalg::GenericOp genericOp) {
  if (genericOp.getNumDpsInits() != 1) {
    return BitWidthChangeInfo::kNull;
  }

  // Check that the all loops are parallel
  unsigned numLoops = genericOp.getNumLoops();
  unsigned numParallelLoops = genericOp.getNumParallelLoops();
  if (numLoops != numParallelLoops) {
    return BitWidthChangeInfo::kNull;
  }

  // Check that all operands that have the highest rank have bit width
  // less than the output bit-width.
  DenseMap<int64_t, SmallVector<OpOperand *>> rankBuckets;
  int64_t maxOperandRank = 0;
  for (OpOperand *input : genericOp.getDpsInputOperands()) {
    auto inputType = dyn_cast<RankedTensorType>(input->get().getType());
    if (!inputType) {
      continue;
    }
    int64_t currRank = inputType.getRank();
    maxOperandRank = std::max(currRank, maxOperandRank);
    rankBuckets[currRank].push_back(input);
  }
  if (maxOperandRank == 0 || rankBuckets[maxOperandRank].empty()) {
    return BitWidthChangeInfo::kNull;
  }

  unsigned int maxInputElementBitWidth = 0;
  OpOperand *inputOperand;
  for (OpOperand *operand : rankBuckets[maxOperandRank]) {
    RankedTensorType tensorType =
        cast<RankedTensorType>(operand->get().getType());
    Type elementType = tensorType.getElementType();
    if (!elementType.isIntOrFloat()) {
      return BitWidthChangeInfo::kNull;
    }
    unsigned elementBitWidth = elementType.getIntOrFloatBitWidth();
    if (elementBitWidth > maxInputElementBitWidth) {
      maxInputElementBitWidth = elementBitWidth;
      inputOperand = operand;
    }
  }
  if (!inputOperand) {
    return BitWidthChangeInfo::kNull;
  }
  Type inputElementType =
      cast<RankedTensorType>(inputOperand->get().getType()).getElementType();

  // Check that the identity input element bitwidth is smaller than the output
  // element bitwidth.
  RankedTensorType outputType =
      dyn_cast<RankedTensorType>(genericOp->getResultTypes()[0]);
  if (!outputType) {
    return BitWidthChangeInfo::kNull;
  }
  Type outputElementType = outputType.getElementType();
  if (!outputElementType.isIntOrFloat()) {
    return BitWidthChangeInfo::kNull;
  }

  unsigned inputBitWidth = inputElementType.getIntOrFloatBitWidth();
  unsigned outputBitWidth = outputElementType.getIntOrFloatBitWidth();

  // Checks specific to bit extend operations.
  if (inputBitWidth < outputBitWidth) {
    // Since these are cloned into dispatches, avoid expensive operations.
    for (Operation &op : *genericOp.getBody()) {
      if (op.getDialect() == op.getContext()->getLoadedDialect("math")) {
        return BitWidthChangeInfo::kNull;
      }
    }
    return BitWidthChangeInfo::kExtend;
  }

  // Checks specific to bit truncate operations.
  if (outputBitWidth < inputBitWidth) {
    // For now enforce that the input and output ranks match for truncates.
    if (maxOperandRank != outputType.getRank()) {
      return BitWidthChangeInfo::kNull;
    }
    return BitWidthChangeInfo::kTruncate;
  }

  return BitWidthChangeInfo::kNull;
}

// Attach BitWidthChangeOpInterface to linalg::GenericOp
struct LinalgGenericBitWidthChange
    : public IREE::LinalgExt::BitWidthChangeOpInterface::ExternalModel<
          LinalgGenericBitWidthChange, linalg::GenericOp> {
  bool isExtensionOp(Operation *op) const {
    return isBitExtendOrTruncateOp(cast<linalg::GenericOp>(op)) ==
           BitWidthChangeInfo::kExtend;
  }

  bool isTruncationOp(Operation *op) const {
    return isBitExtendOrTruncateOp(cast<linalg::GenericOp>(op)) ==
           BitWidthChangeInfo::kTruncate;
  }
};

} // namespace

void registerLinalgExternalModels(DialectRegistry &registry) {
  registry.addExtension(+[](MLIRContext *context,
                            linalg::LinalgDialect *dialect) {
    linalg::GenericOp::attachInterface<LinalgGenericBitWidthChange>(*context);
  });
}

} // namespace mlir::iree_compiler
