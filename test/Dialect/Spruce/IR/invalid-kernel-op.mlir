// RUN: spruce-opt -split-input-file -verify-diagnostics %s

func.func @argument_count(%input_h: index) -> index {
  // expected-error@+1 {{expected one body argument per operand}}
  %result_h = "spruce.kernel"(%input_h) ({
    %c0_d = arith.constant 0 : index
    spruce.yield %c0_d : index
  }) : (index) -> index
  return %result_h : index
}

// -----

func.func @argument_type(%input_h: tensor<16x8xf32>) {
  // expected-error@+1 {{expected body argument 0 to have the same type as operand 0}}
  "spruce.kernel"(%input_h) ({
  ^bb0(%input_d: tensor<8x16xf32>):
    spruce.yield
  }) : (tensor<16x8xf32>) -> ()
  return
}

// -----

func.func @yield_count(%input_h: index) -> index {
  // expected-error@+1 {{expected one yielded value per result}}
  %result_h = spruce.kernel(%input_d = %input_h : index) {
    spruce.yield
  } -> (index)
  return %result_h : index
}

// -----

func.func @yield_type(%input_h: f32) -> index {
  // expected-error@+1 {{expected yield operand 0 to have the same type as result 0}}
  %result_h = spruce.kernel(%input_d = %input_h : f32) {
    spruce.yield %input_d : f32
  } -> (index)
  return %result_h : index
}

// -----

func.func @wrong_terminator(%input_h: index) -> index {
  // expected-error@+1 {{expected body to terminate with spruce.yield}}
  %result_h = spruce.kernel(%input_d = %input_h : index) {
    func.return %input_d : index
  } -> (index)
  return %result_h : index
}

// -----

func.func @implicit_capture(%input_h: index, %host: index) -> index {
  %result_h = spruce.kernel(%input_d = %input_h : index) {
    // expected-error@+1 {{use of undeclared SSA value name}}
    %sum_d = arith.addi %input_d, %host : index
    spruce.yield %sum_d : index
  } -> (index)
  return %result_h : index
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
