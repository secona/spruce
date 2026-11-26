#include "spruce/Spruce.h"

namespace spruce {

void Kernel::compile(CompileTimes &times) {
  times = {};
  const auto start = std::chrono::steady_clock::now();

  // TODO: Compile.

  times.total = std::chrono::duration_cast<std::chrono::nanoseconds>(
      std::chrono::steady_clock::now() - start);
}

void Kernel::run(RunTimes &times) {
  times = {};
  const auto start = std::chrono::steady_clock::now();

  // TODO: Run the kernel.

  times.total = std::chrono::duration_cast<std::chrono::nanoseconds>(
      std::chrono::steady_clock::now() - start);
}

} // namespace spruce
