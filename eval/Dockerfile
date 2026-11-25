FROM nvidia/cuda:13.2.0-devel-ubuntu24.04@sha256:f9492f2eea77fbc3d0c14fa8738f35946b42da72917bf5959d284ca39b4f209a

# ------------------------------------------------------------------------------
# Install required packages
# ------------------------------------------------------------------------------

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        cmake \
        git \
        ninja-build \
        python3-venv

# ------------------------------------------------------------------------------
# Install LLVM 23.1.6
# ------------------------------------------------------------------------------

ARG BUILD_JOBS

RUN git clone --depth 1 --branch llvmorg-23.1.6 \
        https://github.com/llvm/llvm-project.git \
        /src/llvm-project

RUN cmake \
        -S /src/llvm-project/llvm \
        -B /tmp/build \
        -G Ninja \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_INSTALL_PREFIX=/opt/llvm \
        -DLLVM_ENABLE_PROJECTS=mlir \
        -DLLVM_TARGETS_TO_BUILD='X86;NVPTX' \
        -DMLIR_ENABLE_CUDA_RUNNER=ON \
        -DCMAKE_CUDA_ARCHITECTURES=86 \
        -DCMAKE_LIBRARY_PATH=/usr/local/cuda/lib64/stubs \
    && cmake \
        --build /tmp/build \
        --parallel "${BUILD_JOBS:-$(nproc)}" \
        --target install \
    && rm -rf /tmp/build /src/llvm-project

ENV LLVM_DIR=/opt/llvm

# ------------------------------------------------------------------------------
# Prepare Spruce source
# ------------------------------------------------------------------------------

COPY . /src/spruce
RUN git -C /src/spruce submodule update --init --recursive

# ------------------------------------------------------------------------------
# Build
# ------------------------------------------------------------------------------

RUN cmake \
        -S /src/spruce \
        -B /src/spruce/build \
        -G Ninja \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_PREFIX_PATH=/opt/llvm \
        -DBUILD_TESTING=OFF \
    && cmake \
        --build /src/spruce/build \
        --parallel "${BUILD_JOBS:-$(nproc)}" \
        --target spruce-adlet spruce-opt spa-exporter

WORKDIR /eval

CMD ["bash"]
