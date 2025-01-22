hal.executable public @prefill_bs4$async_dispatch_7 {
  hal.executable.variant public @rocm_hsaco_fb target(<"rocm", "rocm-hsaco-fb", {abi = "hip", iree.gpu.target = #iree_gpu.target<arch = "gfx942", features = "", wgp = <compute =  fp64|fp32|fp16|int64|int32|int16|int8, storage =  b64|b32|b16|b8, subgroup =  shuffle|arithmetic, dot =  dp4xi8toi32, mma = [<MFMA_F32_16x16x4_F32>, <MFMA_F32_16x16x16_F16>, <MFMA_F32_32x32x8_F16>, <MFMA_F64_16x16x4_F64>, <MFMA_F32_16x16x16_BF16>, <MFMA_F32_32x32x8_BF16>, <MFMA_F32_16x16x32_F8E5M2FNUZ>, <MFMA_F32_16x16x32_F8E5M2FNUZ_F8E4M3FNUZ>, <MFMA_F32_16x16x32_F8E4M3FNUZ>, <MFMA_F32_16x16x32_F8E4M3FNUZ_F8E5M2FNUZ>, <MFMA_F32_32x32x16_F8E5M2FNUZ>, <MFMA_F32_32x32x16_F8E5M2FNUZ_F8E4M3FNUZ>, <MFMA_F32_32x32x16_F8E4M3FNUZ>, <MFMA_F32_32x32x16_F8E4M3FNUZ_F8E5M2FNUZ>, <MFMA_I32_16x16x32_I8>, <MFMA_I32_32x32x16_I8>], subgroup_size_choices = [64], max_workgroup_sizes = [1024, 1024, 1024], max_thread_count_per_workgroup = 1024, max_workgroup_memory_bytes = 65536, max_workgroup_counts = [2147483647, 2147483647, 2147483647], max_load_instruction_bits = 128, simds_per_wgp = 4, vgpr_space_bits = 16384>>, ukernels = "none"}>) {
    hal.executable.export public @prefill_bs4$async_dispatch_7_scatter_4xDx32x8x128xf16_dispatch_tensor_store ordinal(0) layout(#hal.pipeline.layout<constants = 12, bindings = [#hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>) {
    ^bb0(%arg0: !hal.device, %arg1: index, %arg2: index, %arg3: index, %arg4: index, %arg5: index):
      %x, %y, %z = flow.dispatch.workgroup_count_from_slice %arg1, %arg2, %arg3, %arg4, %arg5
      hal.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @prefill_bs4$async_dispatch_7_scatter_4xDx32x8x128xf16_dispatch_tensor_store() {
        %c32_i64 = arith.constant 32 : i64
        %c0 = arith.constant 0 : index
        %c2 = arith.constant 2 : index
        %c1 = arith.constant 1 : index
        %0 = hal.interface.constant.load layout(<constants = 12, bindings = [#hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>) ordinal(0) : i32
        %1 = hal.interface.constant.load layout(<constants = 12, bindings = [#hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>) ordinal(1) : i32
        %2 = hal.interface.constant.load layout(<constants = 12, bindings = [#hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>) ordinal(2) : i32
        %3 = hal.interface.constant.load layout(<constants = 12, bindings = [#hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>) ordinal(3) : i32
        %4 = hal.interface.constant.load layout(<constants = 12, bindings = [#hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>) ordinal(4) : i32
        %5 = hal.interface.constant.load layout(<constants = 12, bindings = [#hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>) ordinal(5) : i32
        %6 = hal.interface.constant.load layout(<constants = 12, bindings = [#hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>) ordinal(6) : i32
        %7 = hal.interface.constant.load layout(<constants = 12, bindings = [#hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>) ordinal(7) : i32
        %8 = hal.interface.constant.load layout(<constants = 12, bindings = [#hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>) ordinal(8) : i32
        %9 = hal.interface.constant.load layout(<constants = 12, bindings = [#hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>) ordinal(9) : i32
        %10 = hal.interface.constant.load layout(<constants = 12, bindings = [#hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>) ordinal(10) : i32
        %11 = hal.interface.constant.load layout(<constants = 12, bindings = [#hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>) ordinal(11) : i32
        %12 = arith.extui %0 : i32 to i64
        %13 = arith.extui %1 : i32 to i64
        %14 = arith.shli %13, %c32_i64 : i64
        %15 = arith.ori %12, %14 : i64
        %16 = arith.index_castui %15 : i64 to index
        %17 = arith.extui %2 : i32 to i64
        %18 = arith.extui %3 : i32 to i64
        %19 = arith.shli %18, %c32_i64 : i64
        %20 = arith.ori %17, %19 : i64
        %21 = arith.index_castui %20 : i64 to index
        %22 = arith.extui %4 : i32 to i64
        %23 = arith.extui %5 : i32 to i64
        %24 = arith.shli %23, %c32_i64 : i64
        %25 = arith.ori %22, %24 : i64
        %26 = arith.index_castui %25 : i64 to index
        %27 = arith.index_castui %6 : i32 to index
        %28 = arith.index_castui %7 : i32 to index
        %29 = arith.index_castui %8 : i32 to index
        %30 = arith.index_castui %9 : i32 to index
        %31 = arith.extui %10 : i32 to i64
        %32 = arith.extui %11 : i32 to i64
        %33 = arith.shli %32, %c32_i64 : i64
        %34 = arith.ori %31, %33 : i64
        %35 = arith.index_castui %34 : i64 to index
        %36:8 = util.assume.int 
            %16<umin = 4194304, umax = 17175674880>, 
            %21<umin = 4718592, umax = 19322634240>, 
            %26<umin = 4735104, umax = 19389988864>, 
            %27<umin = 32, umax = 131040, udiv = 32>, 
            %28<umin = 32, umax = 131040, udiv = 32>, 
            %29<umin = 1, umax = 4095>, 
            %30<umin = 1, umax = 4095>, 
            %35<umin = 64, umax = 576460752303423424, udiv = 64>
          : index, index, index, index, index, index, index, index
        %37 = flow.dispatch.workload.ordinal %36#4, 1 : index
        %38 = flow.dispatch.workload.ordinal %36#5, 2 : index
        %39 = flow.dispatch.workload.ordinal %36#6, 3 : index
        %40 = flow.dispatch.workload.ordinal %36#7, 4 : index
        %41 = hal.interface.binding.subspan layout(<constants = 12, bindings = [#hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>) binding(0) alignment(64) offset(%36#0) flags("ReadOnly|Indirect") : !flow.dispatch.tensor<readonly:tensor<4x?x8x128xf16>>{%37}
        %42 = hal.interface.binding.subspan layout(<constants = 12, bindings = [#hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>) binding(0) alignment(64) offset(%36#1) flags("ReadOnly|Indirect") : !flow.dispatch.tensor<readonly:tensor<?x32x128xf32>>{%38}
        %43 = hal.interface.binding.subspan layout(<constants = 12, bindings = [#hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>) binding(0) alignment(64) offset(%36#2) flags("ReadOnly|Indirect") : !flow.dispatch.tensor<readonly:tensor<4x?xi32>>{%39}
        %44 = hal.interface.binding.subspan layout(<constants = 12, bindings = [#hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>) binding(1) alignment(64) offset(%c0) flags(Indirect) : !flow.dispatch.tensor<readwrite:tensor<?x32x8x128xf16>>{%40}
        %45 = flow.dispatch.workload.ordinal %36#3, 0 : index
        %46 = flow.dispatch.tensor.load %41, offsets = [0, 0, 0, 0], sizes = [4, %37, 8, 128], strides = [1, 1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<4x?x8x128xf16>>{%37} -> tensor<4x?x8x128xf16>
        %47 = flow.dispatch.tensor.load %42, offsets = [0, 0, 0], sizes = [%38, 32, 128], strides = [1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<?x32x128xf32>>{%38} -> tensor<?x32x128xf32>
        %48 = flow.dispatch.tensor.load %43, offsets = [0, 0], sizes = [4, %39], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<4x?xi32>>{%39} -> tensor<4x?xi32>
        %49 = flow.dispatch.tensor.load %44, offsets = [0, 0, 0, 0], sizes = [%40, 32, 8, 128], strides = [1, 1, 1, 1] : !flow.dispatch.tensor<readwrite:tensor<?x32x8x128xf16>>{%40} -> tensor<?x32x8x128xf16>
        %50 = affine.apply affine_map<()[s0] -> (s0 floordiv 32)>()[%45]
        %51 = tensor.empty(%50) : tensor<4x?x32x8x128xf16>
        %52 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4) -> (d1, d2, d4)>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4)>], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel"]} ins(%47 : tensor<?x32x128xf32>) outs(%51 : tensor<4x?x32x8x128xf16>) {
        ^bb0(%in: f32, %out: f16):
          %54 = linalg.index 0 : index
          %55 = linalg.index 2 : index
          %56 = linalg.index 1 : index
          %57 = affine.apply affine_map<()[s0, s1] -> (s0 + s1 * 32)>()[%55, %56]
          %58 = linalg.index 3 : index
          %59 = linalg.index 4 : index
          %60 = arith.divui %59, %c2 : index
          %61 = arith.remui %59, %c2 : index
          %62 = math.cos %in : f32
          %63 = math.sin %in : f32
          %64 = arith.muli %60, %c2 : index
          %65 = arith.addi %64, %c1 : index
          %extracted = tensor.extract %46[%54, %57, %58, %64] : tensor<4x?x8x128xf16>
          %66 = arith.extf %extracted : f16 to f32
          %extracted_0 = tensor.extract %46[%54, %57, %58, %65] : tensor<4x?x8x128xf16>
          %67 = arith.extf %extracted_0 : f16 to f32
          %68 = arith.cmpi eq, %61, %c0 : index
          %69 = arith.mulf %66, %62 : f32
          %70 = arith.mulf %67, %63 : f32
          %71 = arith.subf %69, %70 : f32
          %72 = arith.mulf %67, %62 : f32
          %73 = arith.mulf %66, %63 : f32
          %74 = arith.addf %72, %73 : f32
          %75 = arith.select %68, %71, %74 : f32
          %76 = arith.truncf %75 : f32 to f16
          linalg.yield %76 : f16
        } -> tensor<4x?x32x8x128xf16>
        %53 = iree_linalg_ext.scatter dimension_map = [0] unique_indices(true) ins(%52, %48 : tensor<4x?x32x8x128xf16>, tensor<4x?xi32>) outs(%49 : tensor<?x32x8x128xf16>) {
        ^bb0(%arg0: f16, %arg1: f16):
          iree_linalg_ext.yield %arg0 : f16
        } -> tensor<?x32x8x128xf16>
        flow.dispatch.tensor.store %53, %44, offsets = [0, 0, 0, 0], sizes = [%40, 32, 8, 128], strides = [1, 1, 1, 1] : tensor<?x32x8x128xf16> -> !flow.dispatch.tensor<readwrite:tensor<?x32x8x128xf16>>{%40}
        return
      }
    }
  }
}
