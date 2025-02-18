// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "compiler/plugins/target/ROCM/builtins/ukernel/common.h"

// Very naive kernel. TODO(bjacob):
// 1. Shared memory: can't allocate it within the microkernel (which is just a
//    helper device function, not the actual amdgpu_kernel). Need to get it
//    passed down here as additional parameters.
// 2. Better scheduling via either barrier intrinsics or inline assemby.
[[clang::always_inline]] void iree_uk_amdgpu_multi_mma_mfma_i32_16x16x32_i8(
    const int8_t *a_buffer, int64_t a_offset, const int8_t *b_buffer,
    int64_t b_offset, int32_t *c_buffer, int64_t c_offset, int32_t k_size,
    int32_t intrinsics_m, int32_t subgroups_m, int32_t intrinsics_n,
    int32_t subgroups_n, int32_t intrinsics_k) {
  // Load existing accumulators. The VLA becomes a normal array after inlining.
  int32x4_t c[intrinsics_m][intrinsics_n];
  int32x4_t *c_global = (int32x4_t *)(c_buffer + c_offset);
  for (int m = 0; m < intrinsics_m; ++m) {
    for (int n = 0; n < intrinsics_n; ++n) {
      c[m][n] = c_global[64 * (m * intrinsics_n + n)];
    }
  }

  // Arithmetic loop.
  const int64_t *a_global = (const int64_t *)(a_buffer + a_offset);
  const int64_t *b_global = (const int64_t *)(b_buffer + b_offset);
  for (int k_outer = 0; k_outer < k_size; ++k_outer) {
    for (int m = 0; m < intrinsics_m; ++m) {
      for (int n = 0; n < intrinsics_n; ++n) {
        for (int k = 0; k < intrinsics_k; ++k) {
          c[m][n] = __builtin_amdgcn_mfma_i32_16x16x32_i8(
              a_global[64 * intrinsics_k * m + k],
              b_global[64 * intrinsics_k * n + k], c[m][n], 0, 0, 0);
        }
      }
    }
    a_global += 64 * intrinsics_m * subgroups_m * intrinsics_k;
    b_global += 64 * intrinsics_n * subgroups_n * intrinsics_k;
  }

  // Store accumulators.
  for (int m = 0; m < intrinsics_m; ++m) {
    for (int n = 0; n < intrinsics_n; ++n) {
      c_global[64 * (m * intrinsics_n + n)] = c[m][n];
    }
  }
}
