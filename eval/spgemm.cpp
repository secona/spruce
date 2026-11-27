#include "spruce/Spruce.h"
#include <iostream>

int main() {
  spruce::InputTensor A("A", "a.mtx");
  A.read();

  spruce::InputTensor B("B", "b.mtx");
  B.read();

  spruce::Kernel kernel("ij,jk->ik", {A, B});

  spruce::Timing t = kernel.compileAndRun();
  t.print();

  spruce::Timing::Aggregator aggregator;
  aggregator.iterate(50, [&kernel] { return kernel.compileAndRun(); });
  aggregator.print();
}
