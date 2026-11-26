#include "spruce/Spruce.h"

int main() {
  spruce::InputTensor A("A", "bcsstk01.mtx");
  A.read();

  spruce::InputTensor B("B", "bcsstk01.mtx");
  B.read();

  spruce::Kernel kernel("ij,jk->ik", {A, B});

  spruce::Timing t = kernel.compile();
  t.print();

  spruce::Timing u = kernel.run();
  u.print();
}
