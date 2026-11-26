#include "spruce/Spruce.h"

#include <iostream>

int main() {
  spruce::InputTensor A("A", "bcsstk01.mtx");
  A.read();

  spruce::InputTensor B("B", "bcsstk01.mtx");
  B.read();

  spruce::Kernel kernel("ij,jk->ik", {A, B});

  spruce::CompileTimes compileTimes;
  kernel.compile(compileTimes);

  std::cout << compileTimes.total.count() << "ns \n";

  spruce::RunTimes runTimes;
  kernel.run(runTimes);
}
