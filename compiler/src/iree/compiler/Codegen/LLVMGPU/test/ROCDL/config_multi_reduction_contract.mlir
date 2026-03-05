// RUN: iree-opt --mlir-print-local-scope --split-input-file --iree-gpu-test-target=gfx1100 \
// RUN:   --pass-pipeline='builtin.module(iree-llvmgpu-select-lowering-strategy)' %s | FileCheck %s

// 6D contraction with 3 reduction dims (fusilli wgrad case).
// f32 inputs on gfx1100 force the scalar SIMT fallback path since gfx1100
// (RDNA3) has no f32-input WMMA/MFMA instructions.
// Verifies that tileK is distributed across ALL reduction dims (d3, d4, d5),
// not just the last one.

#map0 = affine_map<(d0, d1, d2, d3, d4, d5) -> (d3, d4, d5, d0, d2)>
#map1 = affine_map<(d0, d1, d2, d3, d4, d5) -> (d3, d4, d5, d0, d1)>
#map2 = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2)>
func.func @contract_6d_3_reduction_dims(
    %lhs: tensor<16x32x16x8x16xf32>,
    %rhs: tensor<16x32x16x8x32xf32>) -> tensor<8x32x16xf32> {
  %cst = arith.constant 0.000000e+00 : f32
  %empty = tensor.empty() : tensor<8x32x16xf32>
  %fill = linalg.fill ins(%cst : f32) outs(%empty : tensor<8x32x16xf32>) -> tensor<8x32x16xf32>
  %result = linalg.generic {
    indexing_maps = [#map0, #map1, #map2],
    iterator_types = ["parallel", "parallel", "parallel", "reduction", "reduction", "reduction"]}
    ins(%lhs, %rhs : tensor<16x32x16x8x16xf32>, tensor<16x32x16x8x32xf32>)
    outs(%fill : tensor<8x32x16xf32>) {
  ^bb0(%in: f32, %in_0: f32, %out: f32):
    %mul = arith.mulf %in, %in_0 : f32
    %add = arith.addf %mul, %out : f32
    linalg.yield %add : f32
  } -> tensor<8x32x16xf32>
  return %result : tensor<8x32x16xf32>
}

// CHECK-LABEL: func.func @contract_6d_3_reduction_dims(
//  CHECK-SAME:   #iree_codegen.translation_info<pipeline = LLVMGPUTileAndFuse workgroup_size = [32, 8, 1] subgroup_size = 32
//       CHECK:   linalg.generic {{.*}}lowering_config = #iree_gpu.lowering_config
//  CHECK-SAME:     reduction = [0, 0, 0, 1, 2, 16]
//  CHECK-SAME:     thread = [1, 1, 16, 0, 0, 0]
//  CHECK-SAME:     workgroup = [1, 32, 128, 1, 1, 1]

// -----

// 4D contraction with 2 reduction dims (simpler multi-reduction).
// f32 inputs on gfx1100 force the scalar SIMT fallback path.
// Verifies that tileK is distributed across both reduction dims (d2, d3).

#map0 = affine_map<(d0, d1, d2, d3) -> (d2, d3, d0)>
#map1 = affine_map<(d0, d1, d2, d3) -> (d2, d3, d1)>
#map2 = affine_map<(d0, d1, d2, d3) -> (d0, d1)>
func.func @contract_4d_2_reduction_dims(
    %lhs: tensor<32x16x64xf32>,
    %rhs: tensor<32x16x128xf32>) -> tensor<64x128xf32> {
  %cst = arith.constant 0.000000e+00 : f32
  %empty = tensor.empty() : tensor<64x128xf32>
  %fill = linalg.fill ins(%cst : f32) outs(%empty : tensor<64x128xf32>) -> tensor<64x128xf32>
  %result = linalg.generic {
    indexing_maps = [#map0, #map1, #map2],
    iterator_types = ["parallel", "parallel", "reduction", "reduction"]}
    ins(%lhs, %rhs : tensor<32x16x64xf32>, tensor<32x16x128xf32>)
    outs(%fill : tensor<64x128xf32>) {
  ^bb0(%in: f32, %in_0: f32, %out: f32):
    %mul = arith.mulf %in, %in_0 : f32
    %add = arith.addf %mul, %out : f32
    linalg.yield %add : f32
  } -> tensor<64x128xf32>
  return %result : tensor<64x128xf32>
}

// CHECK-LABEL: func.func @contract_4d_2_reduction_dims(
//  CHECK-SAME:   #iree_codegen.translation_info<pipeline = LLVMGPUTileAndFuse workgroup_size = [32, 8, 1] subgroup_size = 32
//       CHECK:   linalg.generic {{.*}}lowering_config = #iree_gpu.lowering_config
//  CHECK-SAME:     reduction = [0, 0, 2, 16]
//  CHECK-SAME:     thread = [1, 16, 0, 0]
//  CHECK-SAME:     workgroup = [32, 128, 1, 1]

// -----

// Standard 3D contraction with 1 reduction dim (regression check).
// f32 inputs on gfx1100 force the scalar SIMT fallback path.
// Verifies single-reduction still works correctly after the multi-reduction fix.

#map0 = affine_map<(d0, d1, d2) -> (d0, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d2, d1)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d1)>
func.func @contract_3d_1_reduction_dim(
    %lhs: tensor<256x512xf32>,
    %rhs: tensor<512x128xf32>) -> tensor<256x128xf32> {
  %cst = arith.constant 0.000000e+00 : f32
  %empty = tensor.empty() : tensor<256x128xf32>
  %fill = linalg.fill ins(%cst : f32) outs(%empty : tensor<256x128xf32>) -> tensor<256x128xf32>
  %result = linalg.generic {
    indexing_maps = [#map0, #map1, #map2],
    iterator_types = ["parallel", "parallel", "reduction"]}
    ins(%lhs, %rhs : tensor<256x512xf32>, tensor<512x128xf32>)
    outs(%fill : tensor<256x128xf32>) {
  ^bb0(%in: f32, %in_0: f32, %out: f32):
    %mul = arith.mulf %in, %in_0 : f32
    %add = arith.addf %mul, %out : f32
    linalg.yield %add : f32
  } -> tensor<256x128xf32>
  return %result : tensor<256x128xf32>
}

// CHECK-LABEL: func.func @contract_3d_1_reduction_dim(
//  CHECK-SAME:   #iree_codegen.translation_info<pipeline = LLVMGPUTileAndFuse workgroup_size = [32, 8, 1] subgroup_size = 32
//       CHECK:   linalg.generic {{.*}}lowering_config = #iree_gpu.lowering_config
//  CHECK-SAME:     reduction = [0, 0, 32]
//  CHECK-SAME:     thread = [1, 16, 0]
//  CHECK-SAME:     workgroup = [32, 128, 1]
