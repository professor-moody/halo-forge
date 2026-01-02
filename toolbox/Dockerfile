# halo-forge Custom Toolbox
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
LABEL version="0.2.0"

# --- base toolchain ---
RUN dnf -y --nodocs --setopt=install_weak_deps=False install \
       make gcc cmake lld clang clang-devel compiler-rt libcurl-devel \
       radeontop git vim patch curl ninja-build tar libatomic xz \
       python python3-devel pip aria2c jq \
       openssh-clients pciutils numactl numactl-devel libdrm mesa-libGL \
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
    tar xzf therock.tar.gz -C /opt/rocm-7.0 --strip-components=1

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
  > /etc/profile.d/rocm.sh && chmod +x /etc/profile.d/rocm.sh && echo 'source /etc/profile.d/rocm.sh' >> /etc/bashrc

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
    transformers \
    peft \
    accelerate \
    safetensors \
    sentencepiece \
    huggingface-hub \
    trl \
    einops \
    datasets \
    scipy \
    protobuf

# --- Training utilities ---
RUN python -m pip install --no-cache-dir \
    tensorboard \
    tqdm \
    pandas \
    matplotlib \
    seaborn \
    pyyaml \
    requests \
    jsonlines \
    paramiko \
    pytest \
    rich

# --- PyTorch memory optimization for Strix Halo (128GB unified memory) ---
ENV PYTORCH_HIP_ALLOC_CONF="backend:native,expandable_segments:True,garbage_collection_threshold:0.9,max_split_size_mb:512"

# Use hipBLASLt for better performance
ENV ROCBLAS_USE_HIPBLASLT=1

# Toolbox compatibility
ENV SHELL=/bin/bash
ENV container=podman

# --- cleanup ---
RUN chmod -R a+rwX /opt && \
    python -m pip cache purge || true && rm -rf /root/.cache/pip || true && \
    dnf clean all && rm -rf /var/cache/dnf/*

WORKDIR /opt/workspace

CMD ["/bin/bash"]
