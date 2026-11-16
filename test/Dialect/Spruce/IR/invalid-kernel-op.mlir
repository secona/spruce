// RUN: spruce-opt -split-input-file -verify-diagnostics %s

func.func @argument_count(%input: index) -> index {
  // expected-error@+1 {{expected one body argument per operand}}
  %result = spruce.kernel(%input) : (index) {
    %c0 = arith.constant 0 : index
    spruce.yield %c0 : index
  } -> (index)
  return %result : index
}

// -----

func.func @argument_type(%input: tensor<16x8xf32>) {
  // expected-error@+1 {{expected body argument 0 to have the same type as operand 0}}
  spruce.kernel(%input) : (tensor<16x8xf32>) {
  ^bb0(%arg: tensor<8x16xf32>):
    spruce.yield
  }
  return
}

// -----

func.func @yield_count(%input: index) -> index {
  // expected-error@+1 {{expected one yielded value per result}}
  %result = spruce.kernel(%input) : (index) {
  ^bb0(%arg: index):
    spruce.yield
  } -> (index)
  return %result : index
}

// -----

func.func @yield_type(%input: f32) -> index {
  // expected-error@+1 {{expected yield operand 0 to have the same type as result 0}}
  %result = spruce.kernel(%input) : (f32) {
  ^bb0(%arg: f32):
    spruce.yield %arg : f32
  } -> (index)
  return %result : index
}

// -----

func.func @wrong_terminator(%input: index) -> index {
  // expected-error@+1 {{expected body to terminate with spruce.yield}}
  %result = spruce.kernel(%input) : (index) {
  ^bb0(%arg: index):
    func.return %arg : index
  } -> (index)
  return %result : index
}

// -----

func.func @implicit_capture(%input: index, %host: index) -> index {
  // expected-note@+1 {{required by region isolation constraints}}
  %result = spruce.kernel(%input) : (index) {
  ^bb0(%arg: index):
    // expected-error@+1 {{using value defined outside the region}}
    %sum = arith.addi %arg, %host : index
    spruce.yield %sum : index
  } -> (index)
  return %result : index
}

// -----

func.func @empty_body() {
  // expected-error@+1 {{region #0 ('body') failed to verify constraint: region with 1 blocks}}
  "spruce.kernel"() ({}) : () -> ()
  return
}

// -----

func.func @multiple_blocks() {
  // expected-error@+1 {{expects region #0 to have 0 or 1 blocks}}
  spruce.kernel() {
    cf.br ^bb1
  ^bb1:
    spruce.yield
  }
  return
}

// -----

func.func @yield_outside_kernel(%input: index) {
  // expected-error@+1 {{expects parent op 'spruce.kernel'}}
  spruce.yield %input : index
}
