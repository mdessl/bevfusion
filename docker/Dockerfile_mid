FROM nvidia/cuda:11.3.1-devel-ubuntu20.04

# Set environment for non-interactive installation
ENV DEBIAN_FRONTEND=noninteractive

# Install all necessary packages in one RUN command
RUN apt-get update && apt-get install -yq \
    wget \
    build-essential \
    g++ \
    gcc \
    libgl1-mesa-glx \
    libglib2.0-0 \
    openmpi-bin \
    openmpi-common \
    libopenmpi-dev \
    libgtk2.0-dev \
    git && \
    rm -rf /var/lib/apt/lists/*

# Install Miniconda
ENV CONDA_DIR=/opt/conda
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh && \
    /bin/bash /tmp/miniconda.sh -b -p $CONDA_DIR && \
    rm /tmp/miniconda.sh

# Update PATH to include conda
ENV PATH=$CONDA_DIR/bin:$PATH

# Install Python and necessary Python packages in a single RUN command
RUN conda install python=3.8 && \
    conda install pytorch==1.10.1 torchvision==0.11.2 torchaudio==0.10.1 cudatoolkit=11.3 -c pytorch && \
    pip install --no-cache-dir Pillow==8.4.0 tqdm torchpack && \
    MMCV_WITH_OPS=1 FORCE_CUDA=1 pip install --no-cache-dir mmcv==2.0.0rc1 -f https://download.openmmlab.com/mmcv/dist/cu113/torch1.10.0/index.html && \
    MMCV_WITH_OPS=1 FORCE_CUDA=1 pip install --no-cache-dir mmcv-full==1.4.0 -f https://download.openmmlab.com/mmcv/dist/cu113/torch1.10.0/index.html && \
    pip install --no-cache-dir mmdet==2.20.0 mpi4py==3.0.3 numba==0.48.0 yapf==0.40.1 setuptools==59.5.0 flash-attn==0.2.0 numpy==1.23.0 nuscenes-devkit && \
    conda clean -afy