#ifndef SPRUCE_ANALYSES_SPARSITYANALYSIS_H
#define SPRUCE_ANALYSES_SPARSITYANALYSIS_H

#include "adlet/graph.hpp"

namespace adlet {

using Graph = ::Graph;
using Tensor = ::Tensor;
using TensorPtr = ::TensorPtr;
using Einsum = ::Einsum;

} // namespace adlet

namespace spruce {

/// These are the mma.sync shapes supported by sm_86 compute capability with
/// f16 input, accumulator, and output.
///
/// https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#warp-level-matrix-shape
enum class MMAShapeKind {
  m8n8k4,
  m16n8k8,
  m16n8k16,
};

void runSparsityAnalysis(std::vector<adlet::TensorPtr> ins);

} // namespace spruce

#endif // SPRUCE_ANALYSES_SPARSITYANALYSIS_H
