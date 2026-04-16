FROM python:3.11-slim

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    MPLCONFIGDIR=/tmp/mplconfig \
    XDG_CACHE_HOME=/tmp/.cache

RUN apt-get update && apt-get install -y --no-install-recommends \
    bash \
    ca-certificates \
    curl \
    git \
    make \
    perl \
    pkg-config \
    build-essential \
    verilator \
    gcc-riscv64-unknown-elf \
    binutils-riscv64-unknown-elf \
    && rm -rf /var/lib/apt/lists/*

RUN python -m pip install --upgrade pip && \
    python -m pip install \
      numpy \
      gymnasium \
      stable-baselines3

WORKDIR /workspace/accelera

# Example:
# docker run --rm -it -v /Users/peter/accelera:/workspace/accelera \
#   --name accelera-rl accelera-rl-daemon:latest \
#   python auto_tuner/rl_daemon.py --timesteps 10000000

