# halo-forge Toolbox
# ROCm 7 nightly + PyTorch for AMD Strix Halo (gfx1151)
#
# Based on kyuz0/amd-strix-halo-llm-finetuning with community improvements
#
# Build: ./build.sh
# Create: toolbox create halo-forge --image localhost/halo-forge:latest
# Enter: toolbox enter halo-forge

FROM registry.fedoraproject.org/fedora:42 AS builder

LABEL maintainer="halo-forge community"
LABEL description="halo-forge: RLVR Training Framework for Strix Halo"
LABEL version="1.0.0"

# --- base toolchain ---
# Includes: compilers, build tools, Python, system utilities
# Multi-language support: MinGW (C++), Rust, Go, .NET
RUN dnf -y --nodocs --setopt=install_weak_deps=False install \
       make gcc gcc-c++ cmake lld clang clang-devel compiler-rt libcurl-devel \
       radeontop git vim patch ninja-build libatomic \
       python python3-devel pip aria2c jq \
       openssh-clients pciutils numactl numactl-devel libdrm mesa-libGL \
       mingw64-gcc-c++ wine \
       rust cargo golang \
       dotnet-sdk-8.0 \
       sshpass \
    && dnf clean all && rm -rf /var/cache/dnf/*

# --- fetch and unpack ROCm TheRock ---
WORKDIR /tmp
ARG ROCM_MAJOR_VER=7
ARG GFX=gfx1151
RUN set -euo pipefail; \
    BASE="https://therock-nightly-tarball.s3.amazonaws.com"; \
    PREFIX="therock-dist-linux-${GFX}-${ROCM_MAJOR_VER}"; \
    KEY="$(curl -s "${BASE}?list-type=2&prefix=${PREFIX}" \
      | grep -o "therock-dist-linux-${GFX}-${ROCM_MAJOR_VER}\.[0-9]\+\.[0-9]\+rc[0-9]\{8\}\.tar\.gz" \
      | sort | tail -n1)"; \
    echo "Latest tarball: ${KEY}"; \
    aria2c -x 16 -s 16 -j 16 --file-allocation=none "${BASE}/${KEY}" -o therock.tar.gz
RUN mkdir -p /opt/rocm-7.0 && \
    tar xzf therock.tar.gz -C /opt/rocm-7.0 --strip-components=1 && \
    rm -f therock.tar.gz

# --- ROCm env vars ---
ENV ROCM_PATH=/opt/rocm-7.0 \
    HIP_PLATFORM=amd \
    HIP_PATH=/opt/rocm-7.0 \
    HIP_CLANG_PATH=/opt/rocm-7.0/llvm/bin \
    HIP_INCLUDE_PATH=/opt/rocm-7.0/include \
    HIP_LIB_PATH=/opt/rocm-7.0/lib \
    HIP_DEVICE_LIB_PATH=/opt/rocm-7.0/lib/llvm/amdgcn/bitcode \
    LD_LIBRARY_PATH=/opt/rocm-7.0/lib:/opt/rocm-7.0/lib64:/opt/rocm-7.0/llvm/lib \
    LIBRARY_PATH=/opt/rocm-7.0/lib:/opt/rocm-7.0/lib64 \
    CPATH=/opt/rocm-7.0/include \
    PKG_CONFIG_PATH=/opt/rocm-7.0/lib/pkgconfig \
    PYTHONNOUSERSITE=1

# Write ROCm env to profile.d for shell sessions
RUN printf '%s\n' \
  'export ROCM_PATH=/opt/rocm-7.0' \
  'export HIP_PLATFORM=amd' \
  'export HIP_PATH=/opt/rocm-7.0' \
  'export HIP_CLANG_PATH=/opt/rocm-7.0/llvm/bin' \
  'export HIP_INCLUDE_PATH=/opt/rocm-7.0/include' \
  'export HIP_LIB_PATH=/opt/rocm-7.0/lib' \
  'export HIP_DEVICE_LIB_PATH=/opt/rocm-7.0/lib/llvm/amdgcn/bitcode' \
  'export LD_LIBRARY_PATH="$HIP_LIB_PATH:$ROCM_PATH/lib:$ROCM_PATH/lib64:$ROCM_PATH/llvm/lib"' \
  'export LIBRARY_PATH="$HIP_LIB_PATH:$ROCM_PATH/lib:$ROCM_PATH/lib64"' \
  'export CPATH="$HIP_INCLUDE_PATH"' \
  'export PKG_CONFIG_PATH="$ROCM_PATH/lib/pkgconfig"' \
  'export ROCBLAS_USE_HIPBLASLT=1' \
  'export PYTHONNOUSERSITE=1' \
  > /etc/profile.d/01-rocm.sh && chmod +x /etc/profile.d/01-rocm.sh

# --- create venv ---
RUN /usr/bin/python3 -m venv /opt/venv
ENV VIRTUAL_ENV=/opt/venv
ENV PATH=/opt/venv/bin:$PATH
RUN python -m pip install --no-cache-dir -U pip setuptools wheel packaging

# --- ROCm PyTorch from AMD nightlies ---
RUN python -m pip install --no-cache-dir \
      --index-url https://rocm.nightlies.amd.com/v2/gfx1151/ \
      --pre torch torchvision

# --- bitsandbytes (upstream with native gfx1151 support) ---
# Using upstream bitsandbytes instead of ROCm fork per:
# https://github.com/kyuz0/amd-strix-halo-llm-finetuning/issues/2
WORKDIR /opt
RUN python -m pip install --no-cache-dir scikit-build-core
RUN git clone https://github.com/bitsandbytes-foundation/bitsandbytes.git
WORKDIR /opt/bitsandbytes
RUN cmake -S . -DGPU_TARGETS="gfx1151" -DBNB_ROCM_ARCH="gfx1151" -DCOMPUTE_BACKEND=hip && \
    make -j && \
    python -m pip install --no-cache-dir . && \
    cd /opt && rm -rf /opt/bitsandbytes

# --- Flash-Attention (ROCm fork) ---
ENV FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE
WORKDIR /opt
RUN git clone https://github.com/ROCm/flash-attention.git && \
    cd flash-attention && \
    python -m pip install --no-cache-dir packaging && \
    git checkout main_perf && \
    python setup.py install && \
    cd /opt && rm -rf /opt/flash-attention

# --- LLM runtime stack ---
RUN python -m pip install --no-cache-dir \
    transformers>=4.46.0 \
    peft>=0.13.0 \
    accelerate \
    safetensors \
    sentencepiece \
    huggingface-hub \
    trl>=0.12.0 \
    einops \
    datasets>=2.14.0 \
    scipy \
    protobuf

# --- Training utilities ---
RUN python -m pip install --no-cache-dir \
    tensorboard \
    wandb \
    tqdm \
    pandas \
    matplotlib \
    seaborn \
    pyyaml>=6.0 \
    requests>=2.31.0 \
    httpx \
    jsonlines \
    paramiko \
    pytest \
    rich

# --- VLM dependencies (Phase 3: Vision-Language) ---
# ultralytics: YOLOv8 for perception verification
# easyocr: Text extraction from images
# pillow: Image processing (usually installed, but ensuring)
RUN python -m pip install --no-cache-dir \
    ultralytics \
    easyocr \
    pillow

# --- Audio dependencies (Phase 4: Audio-Language) ---
# torchaudio: Audio loading and processing (from AMD nightlies to match torch version)
# librosa: Audio feature extraction
# jiwer: Word error rate calculation for ASR
# NOTE: We use torchaudio directly for audio decoding instead of torchcodec,
#       because torchcodec is incompatible with PyTorch 2.11+ nightlies.
RUN python -m pip install --no-cache-dir \
    --index-url https://rocm.nightlies.amd.com/v2/gfx1151/ \
    --pre torchaudio && \
    python -m pip install --no-cache-dir \
    librosa>=0.10.0 \
    jiwer>=3.0.0

# --- Reasoning dependencies (Phase 5: Math/Reasoning) ---
# sympy: Symbolic mathematics for answer verification
RUN python -m pip install --no-cache-dir \
    sympy>=1.12

# --- PyTorch memory optimization for Strix Halo (128GB unified memory) ---
ENV PYTORCH_HIP_ALLOC_CONF="backend:native,expandable_segments:True,garbage_collection_threshold:0.9,max_split_size_mb:512"

# Use hipBLASLt for better performance
ENV ROCBLAS_USE_HIPBLASLT=1

# Toolbox compatibility
ENV SHELL=/bin/bash
ENV container=podman

# --- profile.d scripts ---
COPY scripts/01-triton-env.sh /etc/profile.d/01-triton-env.sh
COPY scripts/99-halo-forge-banner.sh /etc/profile.d/99-halo-forge-banner.sh
COPY scripts/zz-venv-path-fix.sh /etc/profile.d/zz-venv-path-fix.sh

RUN chmod 0644 /etc/profile.d/*.sh && \
    echo 'source /etc/profile.d/01-rocm.sh' >> /etc/bashrc

# --- disable core dumps ---
RUN printf 'ulimit -S -c 0\n' > /etc/profile.d/90-nocoredump.sh && chmod 0644 /etc/profile.d/90-nocoredump.sh

# --- cleanup ---
RUN chmod -R a+rwX /opt && \
    python -m pip cache purge || true && rm -rf /root/.cache/pip || true && \
    dnf clean all && rm -rf /var/cache/dnf/*

WORKDIR /opt/workspace

CMD ["/bin/bash"]
