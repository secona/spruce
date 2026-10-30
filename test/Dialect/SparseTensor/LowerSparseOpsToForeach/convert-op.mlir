// RUN: spruce-opt -lower-sparse-ops-to-foreach -canonicalize %s | FileCheck %s

// CHECK: #[[CSR:.*]] = #sparse_tensor.encoding
#CSR = #sparse_tensor.encoding<{
  map = (i, j) -> (i : dense, j : compressed)
}>

// CHECK-LABEL: func.func @dense_to_csr
// CHECK-SAME:      %[[DENSE:.*]]: tensor<3x4xf64>

// CHECK:         %[[C0:.*]] = arith.constant 0.000000e+00 : f64
// CHECK:         %[[INIT:.*]] = bufferization.alloc_tensor() : tensor<3x4xf64, #[[CSR]]>

// CHECK:         %[[FOREACH:.*]] = sparse_tensor.foreach in %[[DENSE]] init(%[[INIT]])
// CHECK-SAME:       : tensor<3x4xf64>, tensor<3x4xf64, #[[CSR]]> -> tensor<3x4xf64, #[[CSR]]>

// CHECK-SAME:    do {
// CHECK:           ^bb0(%[[ROW:[^:]+]]: index,
// CHECK-SAME:           %[[COL:[^:]+]]: index,
// CHECK-SAME:           %[[VAL:[^:]+]]: f64,
// CHECK-SAME:           %[[ACC:[^:]+]]: tensor<3x4xf64, #[[CSR]]>):
// CHECK:             %[[UNE:.*]] = arith.cmpf une, %[[VAL]], %[[C0]] : f64

// CHECK:             %[[YIELD:.*]] = scf.if %[[UNE]] -> (tensor<3x4xf64, #[[CSR]]>)
// CHECK:               %[[NEW_ACC:.*]] = tensor.insert %[[VAL]] into %[[ACC]][%[[ROW]], %[[COL]]]
// CHECK-SAME:                          : tensor<3x4xf64, #[[CSR]]>
// CHECK:               scf.yield %[[NEW_ACC]] : tensor<3x4xf64, #[[CSR]]>
// CHECK:             else
// CHECK:               scf.yield %[[ACC]] : tensor<3x4xf64, #[[CSR]]>

// CHECK:             sparse_tensor.yield %[[YIELD]] : tensor<3x4xf64, #[[CSR]]>
// CHECK:         }

// CHECK:         %[[SPARSE:.*]] = sparse_tensor.load %[[FOREACH]] hasInserts
// CHECK-SAME:                   : tensor<3x4xf64, #[[CSR]]>
// CHECK:         return %[[SPARSE]] : tensor<3x4xf64, #[[CSR]]>

func.func @dense_to_csr(%dense: tensor<3x4xf64>) -> tensor<3x4xf64, #CSR> {
  %sparse = sparse_tensor.convert %dense
      : tensor<3x4xf64> to tensor<3x4xf64, #CSR>

  return %sparse : tensor<3x4xf64, #CSR>
}
