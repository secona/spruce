// RUN: spruce-opt -split-input-file -verify-diagnostics %s

// expected-error@+2 {{expected valid keyword}}
// expected-error@+1 {{expected a parameter name in struct}}
#missing = #spruce.encoding<{}>

// -----

#not_sparse = 32 : i64
// expected-error@+2 {{invalid kind of attribute specified}}
// expected-error@+1 {{failed to parse Spruce_EncodingAttr parameter 'sparse'}}
#wrong_type = #spruce.encoding<{sparse = #not_sparse}>

// -----

#CSR = #sparse_tensor.encoding<{
  map = (d0, d1) -> (d0 : dense, d1 : compressed)
}>
// expected-error@+1 {{expected '}'}}
#extra_key = #spruce.encoding<{sparse = #CSR, extra = 32 : i64}>

// -----

#CSR = #sparse_tensor.encoding<{
  map = (d0, d1) -> (d0 : dense, d1 : compressed)
}>
#SpruceCSR = #spruce.encoding<{sparse = #CSR}>
// expected-error@+1 {{dimension-rank mismatch between encoding and tensor shape: 2 != 1}}
func.func private @csr_wrong_rank(tensor<?xf32, #SpruceCSR>)

// -----

#BSR = #sparse_tensor.encoding<{
  map = (d0, d1) -> (
    d0 floordiv 2 : dense,
    d1 floordiv 3 : compressed,
    d0 mod 2 : dense,
    d1 mod 3 : dense
  )
}>
#SpruceBSR = #spruce.encoding<{sparse = #BSR}>
// expected-error@+1 {{dimension-rank mismatch between encoding and tensor shape: 2 != 4}}
func.func private @bsr_wrong_rank(tensor<?x?x?x?xf32, #SpruceBSR>)

// -----

#CSR = #sparse_tensor.encoding<{
  map = (d0, d1) -> (d0 : dense, d1 : compressed)
}>
#SpruceCSR = #spruce.encoding<{sparse = #CSR}>
// expected-error@+1 {{invalid primary type}}
func.func private @csr_wrong_element_type(tensor<?x?xi1, #SpruceCSR>)
