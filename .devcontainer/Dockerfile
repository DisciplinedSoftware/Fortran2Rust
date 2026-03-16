FROM mcr.microsoft.com/devcontainers/base:ubuntu-24.04

RUN apt-get update && apt-get install -y \
    build-essential cmake pkg-config curl wget git \
    gfortran f2c \
    libssl-dev zlib1g-dev \
    python3 python3-pip python3-dev python3-venv \
    lsb-release software-properties-common gnupg \
    && rm -rf /var/lib/apt/lists/*

# LLVM/Clang 17 (required by c2rust)
RUN wget https://apt.llvm.org/llvm.sh && chmod +x llvm.sh \
    && ./llvm.sh 17 && rm llvm.sh \
    && apt-get install -y clang-17 libclang-17-dev llvm-17-dev \
    && ln -sf /usr/bin/clang-17 /usr/local/bin/clang \
    && ln -sf /usr/bin/clang++-17 /usr/local/bin/clang++ \
    && rm -rf /var/lib/apt/lists/*

# Rust via rustup (for vscode user)
USER vscode
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain stable
ENV PATH="/home/vscode/.cargo/bin:${PATH}"

USER root
# Make cargo available system-wide
RUN cp -r /home/vscode/.cargo/bin/* /usr/local/bin/ 2>/dev/null || true

# Create a virtual environment for the project (avoids PEP 668 externally-managed error)
RUN python3 -m venv /opt/venv \
    && /opt/venv/bin/pip install --upgrade pip
ENV PATH="/opt/venv/bin:${PATH}"

# Give vscode user write access to the venv so postCreateCommand can install the package
RUN chown -R vscode:vscode /opt/venv
