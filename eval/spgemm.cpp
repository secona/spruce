#include "spruce/Spruce.h"

int main() {
  spruce::InputMLIR src("input_spgemm.mlir");
  src.read();

  spruce::InputTensor A("A", "a.mtx");
  A.read();

  spruce::InputTensor B("B", "b.mtx");
  B.read();

  spruce::Kernel kernel("ij,jk->ik", src, {A, B});

  spruce::Timing t = kernel.compile();
  t.print();

  spruce::Timing::Aggregator aggregator;
  aggregator.iterate(50, [&kernel] { return kernel.run(); });
  aggregator.print();
}
