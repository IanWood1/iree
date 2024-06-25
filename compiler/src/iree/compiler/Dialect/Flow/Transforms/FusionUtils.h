// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//===--- FusionUtils.h --- Utility functions used in fusion ---------------===//
//
// Utility functions to decide of ops are fusable or not, etc.
//
//===----------------------------------------------------------------------===//

#include "iree/compiler/Dialect/LinalgExt/IR/LinalgExtInterfaces.h"
#include "mlir/Dialect/Utils/ReshapeOpsUtils.h"
#include "mlir/IR/PatternMatch.h"

namespace mlir::iree_compiler::IREE::Flow {

/// Return true of the producer and consumer of `operand` are fusable
/// using elementwise op fusion transformation.
bool areFusableAsElementwiseOps(MLIRContext *context, OpOperand *operand,
                                bool fuseMultiReduction);

/// Information needed to expand a `LinalgExt::LinalgFusionOpInterface`
/// operation and fold a reshape with it. This is a more generalized version of
/// `ExpansionInfo` in Linalg/Transforms/ElementwiseOpFusion.cpp that operates
/// on an interface
class ExpansionInfo {
public:
  // Computes the mapping from original dimensions of the op to the dimensions
  // of the expanded op given the `indexingMap` of the fused operand/result of
  // op, the `reassocationMaps` of the reshape op and the shape of
  // the expanded op.
  LogicalResult compute(LinalgExt::LinalgFusionOpInterface fusibleOp,
                        OpOperand *fusableOpOperand,
                        ArrayRef<AffineMap> reassociationMaps,
                        ArrayRef<int64_t> expandedShape,
                        ArrayRef<int64_t> collapsedShape,
                        PatternRewriter &rewriter);
  unsigned getOrigOpNumDims() const;
  unsigned getExpandedOpNumDims() const;
  ReassociationIndicesRef getExpandedDims(unsigned i) const;
  ArrayRef<int64_t> getExpandedShapeOfDim(unsigned i) const;
  ArrayRef<int64_t> getOriginalShape() const;

private:
  /// Reassociation from the dimensions in the original operation to the
  /// dimension of the expanded operation.
  SmallVector<ReassociationIndices> reassociation;
  /// Mapping from extent of loops in the original operation, to the extent of
  /// loops in the expanded operation.
  SmallVector<SmallVector<int64_t>> expandedShapeMap;
  /// Extent of the loop in the original operation.
  SmallVector<int64_t> originalLoopExtent;
  unsigned expandedOpNumDims;
};

/// Expanding the body of an operation requires adaptations of the
/// accessed loop indices. Specifically, access of indices in the original
/// operation need to be replaced with linearizations of indices in the expanded
/// op. That requires the shape of the expanded dimensions to be static (at
/// least all but the most significant). For now check that these are all
/// statically sized. Note that this could be extended to handle dynamic case,
/// but the implementation below uses `affine.apply` which seems to have issues
/// when the shapes are not static.
LogicalResult isOpExpandable(LinalgExt::LinalgFusionOpInterface fusibleOp,
                             const ExpansionInfo &expansionInfo,
                             PatternRewriter &rewriter);

/// Return the indexing map to use in the expanded op for a given the
/// `indexingMap` of the original operation.
AffineMap getIndexingMapInExpandedOp(OpBuilder &builder, AffineMap indexingMap,
                                     const ExpansionInfo &expansionInfo);

/// Return the type of the operand/result to use in the expanded op given
/// the type in the original op.
RankedTensorType getExpandedType(RankedTensorType originalType,
                                 AffineMap indexingMap,
                                 const ExpansionInfo &expansionInfo);

/// Returns the reassociation maps to use in the `tensor.expand_shape`
/// operation to convert the operands of the original operation to operands of
/// the expanded operation. The same method is used to compute the
/// `tensor.collapse_shape` used to collapse the result of the expanded
/// op to get the value that can replace all uses of the results of the original
/// op.
SmallVector<ReassociationIndices>
getReassociationForExpansion(AffineMap indexingMap,
                             const ExpansionInfo &expansionInfo);

LogicalResult
validateDynamicDimExpansion(LinalgExt::LinalgFusionOpInterface interfaceOp,
                            const ExpansionInfo &expansionInfo,
                            PatternRewriter &rewriter);
} // namespace mlir::iree_compiler::IREE::Flow
