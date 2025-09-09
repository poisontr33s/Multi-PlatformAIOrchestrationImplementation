# Multi-Platform AI Orchestration Dockerfile
# Optimized for NVIDIA CUDA 12.2 with multi-model runtime support

FROM nvidia/cuda:12.2-devel-ubuntu22.04

# Prevent timezone prompt during apt install
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=UTC

# Set CUDA environment variables
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=${CUDA_HOME}/bin:${PATH}
ENV LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}

# Install system dependencies
RUN apt-get update && apt-get install -y \
    # Essential build tools
    build-essential \
    cmake \
    git \
    curl \
    wget \
    unzip \
    software-properties-common \
    apt-transport-https \
    ca-certificates \
    gnupg \
    lsb-release \
    # Python development
    python3 \
    python3-pip \
    python3-dev \
    python3-venv \
    # Node.js dependencies
    nodejs \
    npm \
    # System utilities
    htop \
    tmux \
    vim \
    nano \
    jq \
    tree \
    # Network utilities
    netcat \
    telnet \
    # Monitoring tools
    iotop \
    iftop \
    # GPU monitoring
    nvtop \
    # Docker CLI (for Docker-in-Docker)
    docker.io \
    docker-compose \
    # Additional libraries
    libssl-dev \
    libffi-dev \
    libsqlite3-dev \
    libbz2-dev \
    libreadline-dev \
    libncurses5-dev \
    libncursesw5-dev \
    xz-utils \
    tk-dev \
    libxml2-dev \
    libxmlsec1-dev \
    libfuse2 \
    # Cleanup
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install latest Node.js LTS
RUN curl -fsSL https://deb.nodesource.com/setup_lts.x | bash - \
    && apt-get install -y nodejs

# Install GitHub CLI
RUN curl -fsSL https://cli.github.com/packages/githubcli-archive-keyring.gpg | dd of=/usr/share/keyrings/githubcli-archive-keyring.gpg \
    && chmod go+r /usr/share/keyrings/githubcli-archive-keyring.gpg \
    && echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/githubcli-archive-keyring.gpg] https://cli.github.com/packages stable main" | tee /etc/apt/sources.list.d/github-cli.list > /dev/null \
    && apt-get update \
    && apt-get install gh -y

# Create non-root user for development
ARG USERNAME=vscode
ARG USER_UID=1000
ARG USER_GID=$USER_UID

RUN groupadd --gid $USER_GID $USERNAME \
    && useradd --uid $USER_UID --gid $USER_GID -m $USERNAME \
    && usermod -aG docker $USERNAME \
    && echo $USERNAME ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USERNAME \
    && chmod 0440 /etc/sudoers.d/$USERNAME

# Set up workspace directory
RUN mkdir -p /workspace \
    && chown -R $USERNAME:$USERNAME /workspace

# Switch to non-root user
USER $USERNAME
WORKDIR /workspace

# Install Python packages for AI/ML development
RUN python3 -m pip install --user --upgrade pip setuptools wheel

# Install PyTorch with CUDA support
RUN python3 -m pip install --user \
    torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install core AI/ML libraries
RUN python3 -m pip install --user \
    # Transformer models
    transformers==4.36.2 \
    accelerate==0.25.0 \
    bitsandbytes==0.41.3 \
    sentencepiece==0.1.99 \
    protobuf==4.25.1 \
    # Scientific computing
    numpy==1.24.4 \
    pandas==2.1.4 \
    scikit-learn==1.3.2 \
    scipy==1.11.4 \
    # Visualization
    matplotlib==3.8.2 \
    seaborn==0.13.0 \
    plotly==5.17.0 \
    # Jupyter ecosystem
    jupyter==1.0.0 \
    jupyterlab==4.0.9 \
    ipywidgets==8.1.1 \
    # Web framework
    fastapi==0.104.1 \
    uvicorn[standard]==0.24.0 \
    # Data validation
    pydantic==2.5.2 \
    # Database
    sqlalchemy==2.0.23 \
    alembic==1.13.1 \
    # Task queue
    redis==5.0.1 \
    celery==5.3.4 \
    # Testing
    pytest==7.4.3 \
    pytest-asyncio==0.21.1 \
    pytest-cov==4.1.0 \
    # Code quality
    black==23.11.0 \
    flake8==6.1.0 \
    pylint==3.0.3 \
    mypy==1.7.1 \
    pre-commit==3.6.0 \
    # Utilities
    python-dotenv==1.0.0 \
    click==8.1.7 \
    typer==0.9.0 \
    rich==13.7.0 \
    structlog==23.2.0 \
    # Async libraries
    aiohttp==3.9.1 \
    aiofiles==23.2.1 \
    asyncio-mqtt==0.16.1 \
    websockets==12.0 \
    # Monitoring
    prometheus-client==0.19.0 \
    opentelemetry-api==1.21.0 \
    opentelemetry-sdk==1.21.0

# Install cloud provider SDKs
RUN python3 -m pip install --user \
    # Google Cloud
    google-cloud-aiplatform==1.38.1 \
    google-cloud-storage==2.10.0 \
    google-cloud-firestore==2.13.1 \
    google-generativeai==0.3.2 \
    firebase-admin==6.4.0 \
    # Microsoft Azure
    azure-ai-ml==1.11.1 \
    azure-cognitiveservices-language-textanalytics==5.2.0 \
    azure-storage-blob==12.19.0 \
    openai==1.3.8 \
    # AWS (for completeness)
    boto3==1.34.0 \
    botocore==1.34.0

# Install Node.js packages globally
RUN npm install -g \
    typescript@5.3.3 \
    ts-node@10.9.1 \
    @types/node@20.10.5 \
    firebase-tools@12.9.1 \
    @google-cloud/functions-framework@3.3.0 \
    prettier@3.1.1 \
    eslint@8.56.0 \
    @typescript-eslint/parser@6.14.0 \
    @typescript-eslint/eslint-plugin@6.14.0 \
    nodemon@3.0.2 \
    pm2@5.3.0

# Set up Python path
ENV PYTHONPATH=/workspace/src:$PYTHONPATH
ENV PATH=/home/$USERNAME/.local/bin:$PATH

# Create directory structure
RUN mkdir -p \
    /workspace/src \
    /workspace/tests \
    /workspace/configs \
    /workspace/data \
    /workspace/models \
    /workspace/logs \
    /workspace/scripts \
    /workspace/docs \
    /workspace/.cache

# Set up Git configuration template
RUN git config --global init.defaultBranch main \
    && git config --global pull.rebase false \
    && git config --global user.name "AI Orchestration Bot" \
    && git config --global user.email "ai-orchestration@example.com"

# Create health check script
RUN echo '#!/bin/bash\n\
# Health check for the AI orchestration environment\n\
echo "Checking Python environment..."\n\
python3 -c "import torch; print(f\"PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}\")" || exit 1\n\
echo "Checking Node.js environment..."\n\
node --version || exit 1\n\
echo "Checking GPU availability..."\n\
nvidia-smi --query-gpu=name --format=csv,noheader || echo "No GPU detected"\n\
echo "Environment health check passed!"\n\
' > /workspace/health_check.sh && chmod +x /workspace/health_check.sh

# Set default command
CMD ["/bin/bash"]

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD /workspace/health_check.sh

# Labels for metadata
LABEL maintainer="AI Orchestration Team"
LABEL version="1.0.0"
LABEL description="Multi-Platform AI Orchestration Environment with CUDA 12.2"
LABEL ai.orchestration.cuda.version="12.2"
LABEL ai.orchestration.python.version="3.11"
LABEL ai.orchestration.node.version="lts"