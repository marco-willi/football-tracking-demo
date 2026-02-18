FROM pytorch/pytorch:2.9.1-cuda12.6-cudnn9-runtime

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    SHELL=/usr/bin/zsh

# Install system dependencies including zsh
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl wget ca-certificates git build-essential pkg-config \
    ffmpeg libglib2.0-0 libsm6 libxext6 libxrender1 libgl1 \
    make graphviz \
    zsh \
    bash bash-completion \
    && rm -rf /var/lib/apt/lists/*

# Install Oh My Zsh for root user
RUN sh -c "$(curl -fsSL https://raw.githubusercontent.com/ohmyzsh/ohmyzsh/master/tools/install.sh)" "" --unattended

# Set zsh as default shell
RUN chsh -s /usr/bin/zsh

WORKDIR /workspace

# Upgrade pip
RUN pip install --upgrade pip setuptools wheel

# Expose ports
EXPOSE 8888 6006

# Default command
CMD ["/usr/bin/zsh"]
