// Note: this file is for patterns explicitly added only during the
// flow-specific canonicalization pass. Canonicalization patterns registered on
// flow dialect ops should be tested under the appropriate
// iree/compiler/Dialect/Flow/IR/test/*_folding.mlir file for the op category.

// RUN: iree-opt --iree-flow-canonicalize %s --split-input-file --mlir-print-local-scope | FileCheck %s

util.func public @fold_full_insert_into_extract(
    %source: tensor<8x?xf32>,
    %dest: tensor<10x?xf32>,
    %size: index) -> tensor<8x?xf32> {
  %extract = tensor.extract_slice %dest [1, 1] [8, %size] [1, 1] : tensor<10x?xf32> to tensor<8x?xf32>
  %insert = tensor.insert_slice %source into %extract [0, 0] [8, %size] [1, 1] : tensor<8x?xf32> into tensor<8x?xf32>
  util.return %insert : tensor<8x?xf32>
}

// CHECK-LABEL: util.func public @fold_full_insert_into_extract
//  CHECK-SAME:   %[[SOURCE:.+]]: tensor<8x?xf32>
//       CHECK:   util.return %[[SOURCE]]

// -----

util.func public @fold_full_insert_into_empty(
    %source: tensor<8x?xf32>,
    %size: index) -> tensor<8x?xf32> {
  %empty = tensor.empty(%size) : tensor<8x?xf32>
  %insert = tensor.insert_slice %source into %empty [0, 0] [8, %size] [1, 1] : tensor<8x?xf32> into tensor<8x?xf32>
  util.return %insert : tensor<8x?xf32>
}

// CHECK-LABEL: util.func public @fold_full_insert_into_empty
//  CHECK-SAME:   %[[SOURCE:.+]]: tensor<8x?xf32>
//       CHECK:   util.return %[[SOURCE]]

// -----

util.func public @dont_fold_not_full_insert_into_empty(
    %source: tensor<8x?xf32>,
    %size1: index, %size2: index) -> tensor<8x?xf32> {
  %empty = tensor.empty(%size1) : tensor<8x?xf32>
  %insert = tensor.insert_slice %source into %empty [0, 0] [8, %size2] [1, 1] : tensor<8x?xf32> into tensor<8x?xf32>
  util.return %insert : tensor<8x?xf32>
}

// CHECK-LABEL: util.func public @dont_fold_not_full_insert_into_empty
//       CHECK:   %[[INSERT:.+]] = tensor.insert_slice
//       CHECK:   util.return %[[INSERT]]

// -----

util.func public @dont_fold_not_full_static_insert_into_empty(
    %source: tensor<8x?xf32>,
    %size: index) -> tensor<10x?xf32> {
  %empty = tensor.empty(%size) : tensor<10x?xf32>
  %insert = tensor.insert_slice %source into %empty [0, 0] [8, %size] [1, 1] : tensor<8x?xf32> into tensor<10x?xf32>
  util.return %insert : tensor<10x?xf32>
}

// CHECK-LABEL: util.func public @dont_fold_not_full_static_insert_into_empty
//       CHECK:   %[[INSERT:.+]] = tensor.insert_slice
//       CHECK:   util.return %[[INSERT]]

// -----

util.func public @expand_affine(%arg0: index) -> index {
  %mul = affine.apply affine_map<()[s0] -> (s0 * 4)>()[%arg0]
  util.return %mul : index
}

// CHECK-LABEL: util.func public @expand_affine
//  CHECK-SAME:   %[[ARG0:.+]]: index
//       CHECK:   %[[MUL:.+]] = arith.muli %[[ARG0]], %c4 overflow<nsw>
//       CHECK:   util.return %[[MUL]]

// -----

util.func public @decompose_tensor_reshape_2d_to_1d(%input: tensor<4x8xf32>) -> tensor<32xf32> {
  %c32 = arith.constant 32 : i64
  %shape = tensor.from_elements %c32 : tensor<1xi64>
  %reshaped = tensor.reshape %input(%shape) : (tensor<4x8xf32>, tensor<1xi64>) -> tensor<32xf32>
  util.return %reshaped : tensor<32xf32>
}

// CHECK-LABEL: util.func public @decompose_tensor_reshape_2d_to_1d
//  CHECK-SAME:   %[[INPUT:.+]]: tensor<4x8xf32>
//       CHECK:   %[[COLLAPSED:.+]] = tensor.collapse_shape %[[INPUT]] [[0, 1]] : tensor<4x8xf32> into tensor<32xf32>
//       CHECK:   util.return %[[COLLAPSED]]

// -----

util.func public @decompose_tensor_reshape_3d_to_2d(%input: tensor<2x4x8xf32>) -> tensor<8x8xf32> {
  %c8 = arith.constant 8 : i64
  %shape = tensor.from_elements %c8, %c8 : tensor<2xi64>
  %reshaped = tensor.reshape %input(%shape) : (tensor<2x4x8xf32>, tensor<2xi64>) -> tensor<8x8xf32>
  util.return %reshaped : tensor<8x8xf32>
}

// CHECK-LABEL: util.func public @decompose_tensor_reshape_3d_to_2d
//  CHECK-SAME:   %[[INPUT:.+]]: tensor<2x4x8xf32>
//       CHECK:   %[[COLLAPSED:.+]] = tensor.collapse_shape %[[INPUT]] [[0, 1, 2]] : tensor<2x4x8xf32> into tensor<64xf32>
//       CHECK:   %[[EXPANDED:.+]] = tensor.expand_shape %[[COLLAPSED]] [] output_shape [8, 8] : tensor<64xf32> into tensor<8x8xf32>
//       CHECK:   util.return %[[EXPANDED]]

// -----

util.func public @decompose_tensor_reshape_1d_to_2d(%input: tensor<24xf32>) -> tensor<6x4xf32> {
  %c6 = arith.constant 6 : i64
  %c4 = arith.constant 4 : i64
  %shape = tensor.from_elements %c6, %c4 : tensor<2xi64>
  %reshaped = tensor.reshape %input(%shape) : (tensor<24xf32>, tensor<2xi64>) -> tensor<6x4xf32>
  util.return %reshaped : tensor<6x4xf32>
}

// CHECK-LABEL: util.func public @decompose_tensor_reshape_1d_to_2d
//  CHECK-SAME:   %[[INPUT:.+]]: tensor<24xf32>
//       CHECK:   %[[COLLAPSED:.+]] = tensor.collapse_shape %[[INPUT]] [] : tensor<24xf32> into tensor<24xf32>
//       CHECK:   %[[EXPANDED:.+]] = tensor.expand_shape %[[COLLAPSED]] [] output_shape [6, 4] : tensor<24xf32> into tensor<6x4xf32>
//       CHECK:   util.return %[[EXPANDED]]

// -----

util.func public @decompose_tensor_reshape_4d_to_2d(%input: tensor<2x3x4x5xf32>) -> tensor<6x20xf32> {
  %c6 = arith.constant 6 : i64
  %c20 = arith.constant 20 : i64
  %shape = tensor.from_elements %c6, %c20 : tensor<2xi64>
  %reshaped = tensor.reshape %input(%shape) : (tensor<2x3x4x5xf32>, tensor<2xi64>) -> tensor<6x20xf32>
  util.return %reshaped : tensor<6x20xf32>
}

// CHECK-LABEL: util.func public @decompose_tensor_reshape_4d_to_2d
//  CHECK-SAME:   %[[INPUT:.+]]: tensor<2x3x4x5xf32>
//       CHECK:   %[[COLLAPSED:.+]] = tensor.collapse_shape %[[INPUT]] [[0, 1, 2, 3]] : tensor<2x3x4x5xf32> into tensor<120xf32>
//       CHECK:   %[[EXPANDED:.+]] = tensor.expand_shape %[[COLLAPSED]] [] output_shape [6, 20] : tensor<120xf32> into tensor<6x20xf32>
//       CHECK:   util.return %[[EXPANDED]]

// -----

util.func public @decompose_tensor_reshape_with_dynamic_dims(%input: tensor<?x8xf32>, %size: index) -> tensor<?xf32> {
  %sz_i64 = arith.index_cast %size : index to i64
  %shape = tensor.from_elements %sz_i64 : tensor<1xi64>
  %reshaped = tensor.reshape %input(%shape) : (tensor<?x8xf32>, tensor<1xi64>) -> tensor<?xf32>
  util.return %reshaped : tensor<?xf32>
}

// CHECK-LABEL: util.func public @decompose_tensor_reshape_with_dynamic_dims
//  CHECK-SAME:   %[[INPUT:.+]]: tensor<?x8xf32>
//       CHECK:   %[[COLLAPSED:.+]] = tensor.collapse_shape %[[INPUT]] [[0, 1]] : tensor<?x8xf32> into tensor<?xf32>
//       CHECK:   util.return %[[COLLAPSED]]

// -----

util.func public @decompose_tensor_reshape_same_shape(%input: tensor<4x8xf32>) -> tensor<4x8xf32> {
  %c4 = arith.constant 4 : i64
  %c8 = arith.constant 8 : i64
  %shape = tensor.from_elements %c4, %c8 : tensor<2xi64>
  %reshaped = tensor.reshape %input(%shape) : (tensor<4x8xf32>, tensor<2xi64>) -> tensor<4x8xf32>
  util.return %reshaped : tensor<4x8xf32>
}

// CHECK-LABEL: util.func public @decompose_tensor_reshape_same_shape
//  CHECK-SAME:   %[[INPUT:.+]]: tensor<4x8xf32>
//       CHECK:   %[[COLLAPSED:.+]] = tensor.collapse_shape %[[INPUT]] [[0, 1]] : tensor<4x8xf32> into tensor<32xf32>
//       CHECK:   %[[EXPANDED:.+]] = tensor.expand_shape %[[COLLAPSED]] [] output_shape [4, 8] : tensor<32xf32> into tensor<4x8xf32>
//       CHECK:   util.return %[[EXPANDED]]
