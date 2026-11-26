#include "spruce/Spruce.h"

int main() {
  spruce::Tensor A("A", "bcsstk01.mtx");
  spruce::Tensor B("B", "bcsstk01.mtx");

  spruce::Kernel kernel("ij,jk->ik", {A, B});

  spruce::CompileTimes compileTimes;
  kernel.compile(compileTimes);

  kernel.run();
}
