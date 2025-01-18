// RUN: iree-opt --split-input-file --mlir-print-local-scope --pass-pipeline="builtin.module(util.func(iree-dispatch-creation-bubble-up-expand-shapes))" %s | FileCheck %s

util.func public @bubbble_expand_through_extract(%arg0 : tensor<2x4096x5120xf16>) -> (tensor<2x64x64x2560xf16>) {
  %extracted_slice_237 = tensor.extract_slice %arg0[0, 0, 0] [2, 4096, 2560] [1, 1, 1] : tensor<2x4096x5120xf16> to tensor<2x4096x2560xf16>
  %expanded_239 = tensor.expand_shape %extracted_slice_237 [[0], [1, 2], [3]] output_shape [2, 64, 64, 2560] : tensor<2x4096x2560xf16> into tensor<2x64x64x2560xf16>
  util.return %expanded_239 : tensor<2x64x64x2560xf16>
}

// CHECK-LABEL:  @bubbble_expand_through_extract
//       CHECK:    %[[EXPAND:.+]] = tensor.expand_shape
//       CHECK:    %[[EXTRACT:.+]] = tensor.extract_slice %[[EXPAND]]

// -----

util.func public @unsupported_bubbble_expand_through_extract(%arg0 : tensor<2x4096x5120xf16>) -> (tensor<2x32x64x2560xf16>) {
  %extracted_slice_237 = tensor.extract_slice %arg0[0, 0, 0] [2, 2048, 2560] [1, 1, 1] : tensor<2x4096x5120xf16> to tensor<2x2048x2560xf16>
  %expanded_239 = tensor.expand_shape %extracted_slice_237 [[0], [1, 2], [3]] output_shape [2, 32, 64, 2560] : tensor<2x2048x2560xf16> into tensor<2x32x64x2560xf16>
  util.return %expanded_239 : tensor<2x32x64x2560xf16>
}

// CHECK-LABEL:  @unsupported_bubbble_expand_through_extract
//       CHECK:    %[[EXTRACT:.+]] = tensor.extract_slice
//       CHECK:    %[[EXPAND:.+]] = tensor.expand_shape %[[EXTRACT]]

// -----

// Checks two things
// 1. Propagation of reshapes across attention operations
// 2. Use of folders to convert (expand(collapse)) -> (collapse)
util.func public @attention_v_reshape_propagation(%arg0: index,
    %arg1: tensor<4x8x4x128x?xf16>, %arg2: tensor<128x?x128xf16>,
    %arg3: tensor<128x?x128xf16>, %arg4: f16, %arg5: tensor<128x?x?xf16>)
    -> tensor<4x?x32x128xf16> {
  %0 = tensor.empty(%arg0) : tensor<4x?x32x128xf16>
  %1 = tensor.empty(%arg0) : tensor<128x?x128xf16>
  %collapsed = tensor.collapse_shape %arg1 [[0, 1, 2], [3], [4]]
      : tensor<4x8x4x128x?xf16> into tensor<128x128x?xf16>
  %4 = iree_linalg_ext.attention {
      indexing_maps = [affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d3)>,
                       affine_map<(d0, d1, d2, d3, d4) -> (d0, d4, d3)>,
                       affine_map<(d0, d1, d2, d3, d4) -> (d0, d2, d4)>,
                       affine_map<(d0, d1, d2, d3, d4) -> ()>,
                       affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d4)>,
                       affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2)>]}
      ins(%arg2, %arg3, %collapsed, %arg4, %arg5
        : tensor<128x?x128xf16>, tensor<128x?x128xf16>, tensor<128x128x?xf16>,
          f16, tensor<128x?x?xf16>)
      outs(%1 : tensor<128x?x128xf16>) {
      ^bb0(%arg6: f32):
    iree_linalg_ext.yield %arg6 : f32
  } -> tensor<128x?x128xf16>
  %expanded = tensor.expand_shape %4 [[0, 1], [2], [3]]
      output_shape [4, 32, %arg0, 128]
      : tensor<128x?x128xf16> into tensor<4x32x?x128xf16>
  %5 = linalg.generic {
      indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                       affine_map<(d0, d1, d2, d3) -> (d0, d2, d1, d3)>],
      iterator_types = ["parallel", "parallel", "parallel", "parallel"]}
      ins(%expanded : tensor<4x32x?x128xf16>) outs(%0 : tensor<4x?x32x128xf16>) {                                                                                                                                        ^bb0(%in: f16, %out: f16):
    linalg.yield %in : f16
  } -> tensor<4x?x32x128xf16>
  util.return %5 : tensor<4x?x32x128xf16>
}
// CHECK-LABEL: func public @attention_v_reshape_propagation
//  CHECK-SAME:     %[[ARG1:[a-zA-Z0-9]+]]: tensor<4x8x4x128x?xf16>
//       CHECK:   %[[ATTENTION:.+]] = iree_linalg_ext.attention
//  CHECK-SAME:       ins(%{{.+}}, %{{.+}}, %[[ARG1]],
//       CHECK:   %[[GENERIC:.+]] = linalg.generic
//  CHECK-SAME:       ins(%[[ATTENTION]]
//       CHECK:   %[[COLLAPSE:.+]] = tensor.collapse_shape %[[GENERIC]]
//       CHECK:   return %[[COLLAPSE]]

// -----

// This can get folded to a single expand shape.
util.func @ex(%arg0: tensor<4x?x8x128xf16>, %arg1: index) -> tensor<4x?x32x8x128xf16> {
  %collapsed = tensor.collapse_shape %arg0 [[0, 1], [2, 3]] : tensor<4x?x8x128xf16> into tensor<?x1024xf16>
  %expanded = tensor.expand_shape %collapsed [[0, 1, 2], [3, 4]] output_shape [4, %arg1, 32, 8, 128] : tensor<?x1024xf16> into tensor<4x?x32x8x128xf16>
  util.return %expanded : tensor<4x?x32x8x128xf16>
}

// -----

// This requires "bubbling".
util.func @ex0(%arg0: tensor<4x?x8x128xf16>, %arg1: index) -> tensor<?x32x8x128xf16> {
  %collapsed = tensor.collapse_shape %arg0 [[0, 1], [2, 3]] : tensor<4x?x8x128xf16> into tensor<?x1024xf16>
  %expanded = tensor.expand_shape %collapsed [[0, 1], [2, 3]] output_shape [%arg1, 32, 8, 128] : tensor<?x1024xf16> into tensor<?x32x8x128xf16>
  util.return %expanded : tensor<?x32x8x128xf16>
}

// -----

util.func @ex1(%arg0: tensor<4x?x8x128xf16>, %arg1: index) -> tensor<?x32x8x128xf16> {
  %collapsed = tensor.collapse_shape %arg0 [[0, 1], [2], [3]] : tensor<4x?x8x128xf16> into tensor<?x8x128xf16>
  %expanded = tensor.expand_shape %collapsed [[0, 1], [2], [3]] output_shape [%arg1, 32, 8, 128] : tensor<?x8x128xf16> into tensor<?x32x8x128xf16>
  util.return %expanded : tensor<?x32x8x128xf16>
}

// -----

util.func @ex2(%arg0: tensor<?x?x8x128xf16>, %arg1: index, %arg2: index) -> tensor<?x32x8x128xf16> {
  %collapsed = tensor.collapse_shape %arg0 [[0, 1], [2], [3]] : tensor<?x?x8x128xf16> into tensor<?x8x128xf16>
  %expanded = tensor.expand_shape %collapsed [[0, 1], [2], [3]] output_shape [%arg1, %arg2, 8, 128] : tensor<?x8x128xf16> into tensor<?x32x8x128xf16>
  util.return %expanded : tensor<?x32x8x128xf16>
}

// -----

// This requires "bubbling".
util.func @ex3(%arg0: tensor<4x?x8x128xf16>, %arg1: index) -> tensor<?x32x8x128xf16> {
  %collapsed = tensor.collapse_shape %arg0 [[0, 1, 2], [3]] : tensor<4x?x8x128xf16> into tensor<?x128xf16>
  %expanded = tensor.expand_shape %collapsed [[0, 1, 2], [3]] output_shape [%arg1, 32, 8, 128] : tensor<?x128xf16> into tensor<?x32x8x128xf16>
  util.return %expanded : tensor<?x32x8x128xf16>
}

// -----

// This requires "bubbling".
util.func @ex4(%arg0: tensor<4x?x8x128xf16>, %arg1: index) -> tensor<?x2x8x128xf16> {
  %collapsed = tensor.collapse_shape %arg0 [[0, 1, 2], [3]] : tensor<4x?x8x128xf16> into tensor<?x128xf16>
  %expanded = tensor.expand_shape %collapsed [[0, 1, 2], [3]] output_shape [%arg1, 2, 8, 128] : tensor<?x128xf16> into tensor<?x2x8x128xf16>
  util.return %expanded : tensor<?x2x8x128xf16>
}
