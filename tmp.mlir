#map = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#map1 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3, d2)>
util.func public @elementwise_dag_transpose(%arg0: tensor<?x?x?x?xf32>, %empty: tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32> {
  %cst = arith.constant 3.14 : f32
  // Check that reducing dims propagates more than 1 op away
  %elementwise0 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg0 : tensor<?x?x?x?xf32>) outs(%empty : tensor<?x?x?x?xf32>) {
  ^bb0(%in : f32, %out : f32):
    %22 = arith.mulf %cst, %in : f32
    linalg.yield %22 : f32
  } -> tensor<?x?x?x?xf32>
  %elementwise1 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%elementwise0: tensor<?x?x?x?xf32>) outs(%empty : tensor<?x?x?x?xf32>) {
  ^bb0(%in : f32, %out : f32):
    %22 = arith.mulf %cst, %in : f32
    linalg.yield %22 : f32
  } -> tensor<?x?x?x?xf32>
  %elementwise2 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%elementwise1 : tensor<?x?x?x?xf32>) outs(%empty : tensor<?x?x?x?xf32>) {
  ^bb0(%in : f32, %out : f32):
    %22 = arith.mulf %cst, %in : f32
    linalg.yield %22 : f32
  } -> tensor<?x?x?x?xf32>
  %elementwise3 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%elementwise1 : tensor<?x?x?x?xf32>) outs(%empty : tensor<?x?x?x?xf32>) {
  ^bb0(%in : f32, %out : f32):
    %22 = arith.mulf %cst, %in : f32
    linalg.yield %22 : f32
  } -> tensor<?x?x?x?xf32>
  %elementwise4 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%elementwise3, %elementwise2 : tensor<?x?x?x?xf32>, tensor<?x?x?x?xf32>) outs(%empty : tensor<?x?x?x?xf32>) {
  ^bb0(%in : f32, %in_1 : f32, %out : f32):
    %22 = arith.mulf %in_1, %in : f32
    linalg.yield %22 : f32
  } -> tensor<?x?x?x?xf32>

  // Check that reducing dims propagates more than 1 op away
  %elementwise5 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%elementwise4 : tensor<?x?x?x?xf32>) outs(%empty : tensor<?x?x?x?xf32>) {
  ^bb0(%in : f32, %out : f32):
    %22 = arith.mulf %cst, %in : f32
    linalg.yield %22 : f32
  } -> tensor<?x?x?x?xf32>

  util.return %elementwise5 : tensor<?x?x?x?xf32>
}
