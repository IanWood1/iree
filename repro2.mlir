hal.executable public @prefill_bs4$async_dispatch_9 {
  hal.executable.variant public @rocm_hsaco_fb target(<"rocm", "rocm-hsaco-fb", {abi = "hip", iree.gpu.target = #iree_gpu.target<arch = "gfx942", features = "", wgp = <compute =  fp64|fp32|fp16|int64|int32|int16|int8, storage =  b64|b32|b16|b8, subgroup =  shuffle|arithmetic, dot =  dp4xi8toi32, mma = [<MFMA_F32_16x16x4_F32>, <MFMA_F32_16x16x16_F16>, <MFMA_F32_32x32x8_F16>, <MFMA_F64_16x16x4_F64>, <MFMA_F32_16x16x16_BF16>, <MFMA_F32_32x32x8_BF16>, <MFMA_F32_16x16x32_F8E5M2FNUZ>, <MFMA_F32_16x16x32_F8E5M2FNUZ_F8E4M3FNUZ>, <MFMA_F32_16x16x32_F8E4M3FNUZ>, <MFMA_F32_16x16x32_F8E4M3FNUZ_F8E5M2FNUZ>, <MFMA_F32_32x32x16_F8E5M2FNUZ>, <MFMA_F32_32x32x16_F8E5M2FNUZ_F8E4M3FNUZ>, <MFMA_F32_32x32x16_F8E4M3FNUZ>, <MFMA_F32_32x32x16_F8E4M3FNUZ_F8E5M2FNUZ>, <MFMA_I32_16x16x32_I8>, <MFMA_I32_32x32x16_I8>], subgroup_size_choices = [64], max_workgroup_sizes = [1024, 1024, 1024], max_thread_count_per_workgroup = 1024, max_workgroup_memory_bytes = 65536, max_workgroup_counts = [2147483647, 2147483647, 2147483647], max_load_instruction_bits = 128, simds_per_wgp = 4, vgpr_space_bits = 16384>>, ukernels = "none"}>) {
    hal.executable.export public @prefill_bs4$async_dispatch_9_scatter_4xDx32x8x128xf16_dispatch_tensor_store ordinal(0) layout(#hal.pipeline.layout<constants = 8, bindings = [#hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>) {
    ^bb0(%arg0: !hal.device, %arg1: index, %arg2: index, %arg3: index):
      %x, %y, %z = flow.dispatch.workgroup_count_from_slice %arg1, %arg2, %arg3
      hal.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @prefill_bs4$async_dispatch_9_scatter_4xDx32x8x128xf16_dispatch_tensor_store() {
        %c32_i64 = arith.constant 32 : i64
        %c0 = arith.constant 0 : index
        %0 = hal.interface.constant.load layout(<constants = 8, bindings = [#hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>) ordinal(0) : i32
        %1 = hal.interface.constant.load layout(<constants = 8, bindings = [#hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>) ordinal(1) : i32
        %2 = hal.interface.constant.load layout(<constants = 8, bindings = [#hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>) ordinal(2) : i32
        %3 = hal.interface.constant.load layout(<constants = 8, bindings = [#hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>) ordinal(3) : i32
        %4 = hal.interface.constant.load layout(<constants = 8, bindings = [#hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>) ordinal(4) : i32
        %5 = hal.interface.constant.load layout(<constants = 8, bindings = [#hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>) ordinal(5) : i32
        %6 = hal.interface.constant.load layout(<constants = 8, bindings = [#hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>) ordinal(6) : i32
        %7 = hal.interface.constant.load layout(<constants = 8, bindings = [#hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>) ordinal(7) : i32
        %8 = arith.extui %0 : i32 to i64
        %9 = arith.extui %1 : i32 to i64
        %10 = arith.shli %9, %c32_i64 : i64
        %11 = arith.ori %8, %10 : i64
        %12 = arith.index_castui %11 : i64 to index
        %13 = arith.extui %2 : i32 to i64
        %14 = arith.extui %3 : i32 to i64
        %15 = arith.shli %14, %c32_i64 : i64
        %16 = arith.ori %13, %15 : i64
        %17 = arith.index_castui %16 : i64 to index
        %18 = arith.index_castui %4 : i32 to index
        %19 = arith.index_castui %5 : i32 to index
        %20 = arith.extui %6 : i32 to i64
        %21 = arith.extui %7 : i32 to i64
        %22 = arith.shli %21, %c32_i64 : i64
        %23 = arith.ori %20, %22 : i64
        %24 = arith.index_castui %23 : i64 to index
        %25:5 = util.assume.int 
            %12<umin = 4456448, umax = 18249154560>, 
            %17<umin = 4735168, umax = 19390054400>, 
            %18<umin = 1, umax = 4095>, 
            %19<umin = 1, umax = 4095>, 
            %24<umin = 64, umax = 576460752303423424, udiv = 64>
          : index, index, index, index, index
        %26 = flow.dispatch.workload.ordinal %25#2, 0 : index
        %27 = flow.dispatch.workload.ordinal %25#3, 1 : index
        %28 = flow.dispatch.workload.ordinal %25#4, 2 : index
        %29 = hal.interface.binding.subspan layout(<constants = 8, bindings = [#hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>) binding(0) alignment(64) offset(%25#0) flags("ReadOnly|Indirect") : !flow.dispatch.tensor<readonly:tensor<4x?x32x8x128xf16>>{%26}
        %30 = hal.interface.binding.subspan layout(<constants = 8, bindings = [#hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>) binding(0) alignment(64) offset(%25#1) flags("ReadOnly|Indirect") : !flow.dispatch.tensor<readonly:tensor<4x?xi32>>{%27}
        %31 = hal.interface.binding.subspan layout(<constants = 8, bindings = [#hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>) binding(1) alignment(64) offset(%c0) flags(Indirect) : !flow.dispatch.tensor<readwrite:tensor<?x32x8x128xf16>>{%28}
        %32 = flow.dispatch.tensor.load %29, offsets = [0, 0, 0, 0, 0], sizes = [4, %26, 32, 8, 128], strides = [1, 1, 1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<4x?x32x8x128xf16>>{%26} -> tensor<4x?x32x8x128xf16>
        %33 = flow.dispatch.tensor.load %30, offsets = [0, 0], sizes = [4, %27], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<4x?xi32>>{%27} -> tensor<4x?xi32>
        %34 = flow.dispatch.tensor.load %31, offsets = [0, 0, 0, 0], sizes = [%28, 32, 8, 128], strides = [1, 1, 1, 1] : !flow.dispatch.tensor<readwrite:tensor<?x32x8x128xf16>>{%28} -> tensor<?x32x8x128xf16>
        %35 = iree_linalg_ext.scatter dimension_map = [0] unique_indices(true) ins(%32, %33 : tensor<4x?x32x8x128xf16>, tensor<4x?xi32>) outs(%34 : tensor<?x32x8x128xf16>) {
        ^bb0(%arg0: f16, %arg1: f16):
          iree_linalg_ext.yield %arg0 : f16
        } -> tensor<?x32x8x128xf16>
        flow.dispatch.tensor.store %35, %31, offsets = [0, 0, 0, 0], sizes = [%28, 32, 8, 128], strides = [1, 1, 1, 1] : tensor<?x32x8x128xf16> -> !flow.dispatch.tensor<readwrite:tensor<?x32x8x128xf16>>{%28}
        return
      }
    }
  }
}
