# halo-forge Custom Toolbox
# ROCm 7 nightly + PyTorch 2.7 for AMD Strix Halo (gfx1151)
#
# Build: ./build.sh
# Create: toolbox create halo-forge --image localhost/halo-forge:latest
# Enter: toolbox enter halo-forge

FROM fedora:42

LABEL maintainer="keys"
LABEL description="halo-forge: RLVR Training Framework for Strix Halo"
LABEL version="0.1.0"

# Install base dependencies including Python 3.11
RUN dnf update -y && dnf install -y \
    python3.11 \
    python3.11-devel \
    git \
    wget \
    curl \
    vim \
    htop \
    ninja-build \
    cmake \
    gcc \
    gcc-c++ \
    make \
    pciutils \
    numactl \
    numactl-devel \
    libdrm \
    mesa-libGL \
    openssh-clients \
    && dnf clean all

# Create venv with Python 3.11
WORKDIR /opt
RUN python3.11 -m ensurepip && python3.11 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
ENV VIRTUAL_ENV="/opt/venv"

# Install PyTorch with ROCm support from scottt's wheel (gfx1151 optimized)
# PyTorch 2.7.0a0 built against ROCm 6.5
RUN pip install --no-cache-dir \
    https://github.com/scottt/rocm-TheRock/releases/download/v6.5.0rc-pytorch/torch-2.7.0a0+gitbfd8155-cp311-cp311-linux_x86_64.whl

# Install ML libraries
RUN pip install --no-cache-dir \
    transformers>=4.46.0 \
    datasets \
    accelerate \
    peft>=0.13.0 \
    trl>=0.12.0 \
    bitsandbytes \
    scipy \
    sentencepiece \
    protobuf \
    einops

# Install training utilities
RUN pip install --no-cache-dir \
    tensorboard \
    wandb \
    tqdm \
    pandas \
    matplotlib \
    pyyaml \
    requests \
    jsonlines

# PyTorch memory optimization for Strix Halo (96GB unified memory)
ENV PYTORCH_HIP_ALLOC_CONF="backend:native,expandable_segments:True,garbage_collection_threshold:0.9,max_split_size_mb:512"

# Use hipBLASLt for better performance
ENV ROCBLAS_USE_HIPBLASLT=1

# Toolbox compatibility
ENV SHELL=/bin/bash
ENV container=podman

WORKDIR /opt/workspace

# Entrypoint activates venv
RUN printf '#!/bin/bash\nsource /opt/venv/bin/activate\nexec "$@"\n' > /entrypoint.sh && \
    chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]
CMD ["/bin/bash"]

