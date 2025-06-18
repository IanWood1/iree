func.func @concat0(%arg0 : tensor<128x64xf32>, %arg1 : tensor<128x64xf32>) -> tensor<128xf32> {
  %dispatch = flow.dispatch.region -> (tensor<128xf32>) {
    %0 = tensor.concat dim(1) %arg0, %arg1 : (tensor<128x64xf32>, tensor<128x64xf32>) -> tensor<128x128xf32>
    %14 = tensor.empty() : tensor<128xf32>
    %cst0 = arith.constant 0.0 : f32 
    %15 = linalg.fill ins(%cst0 : f32) outs(%14 : tensor<128xf32>) -> tensor<128xf32>
    %1 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0)>], iterator_types = ["parallel", "reduction"]} ins(%0 : tensor<128x128xf32>) outs(%15 : tensor<128xf32>) {
    ^bb0(%in: f32, %out: f32):
      %19 = arith.addf %in, %out : f32
      linalg.yield %19 : f32
    } -> tensor<128xf32>
    flow.return %1 : tensor<128xf32>
  }
  return %dispatch : tensor<128xf32>
}

func.func @concat1(%arg0 : tensor<128x64xf32>, %arg1 : tensor<128x64xf32>) -> tensor<128xf32> {
  %dispatch = flow.dispatch.region -> (tensor<128xf32>) {
    %lhs = linalg.floor ins(%arg0 : tensor<128x64xf32>) outs(%arg0 : tensor<128x64xf32>) -> tensor<128x64xf32>
    %rhs = linalg.floor ins(%arg1 : tensor<128x64xf32>) outs(%arg1 : tensor<128x64xf32>) -> tensor<128x64xf32>
    %0 = tensor.concat dim(1) %lhs, %rhs : (tensor<128x64xf32>, tensor<128x64xf32>) -> tensor<128x128xf32>
    %14 = tensor.empty() : tensor<128xf32>
    %cst0 = arith.constant 0.0 : f32 
    %15 = linalg.fill ins(%cst0 : f32) outs(%14 : tensor<128xf32>) -> tensor<128xf32>
    %1 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0)>], iterator_types = ["parallel", "reduction"]} ins(%0 : tensor<128x128xf32>) outs(%15 : tensor<128xf32>) {
    ^bb0(%in: f32, %out: f32):
      %19 = arith.addf %in, %out : f32
      linalg.yield %19 : f32
    } -> tensor<128xf32>
    flow.return %1 : tensor<128xf32>
  }
  return %dispatch : tensor<128xf32>
}

