FROM nvidia/cuda:10.0-devel

RUN apt-get update

RUN apt-get update && apt-get install --no-install-recommends -qqy \
    linux-headers-$(uname -r) \
    build-essential \
    curl \
    libfreetype6-dev \
    libhdf5-serial-dev \
    libpng-dev \
    libzmq3-dev \
    pkg-config \
    python3 \
    python3-dev \
    rsync \
    software-properties-common \
    unzip \
    wget \
    vim \
    git \
    gnupg2 \
    x11-apps \
    libsm6 \
    libxext6 \
    libxrender1 && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*
    
# Install development and runtime libraries (~4GB)
RUN DEBIAN_FRONTEND=noninteractive \
    apt-get update && \
    apt-get install --no-install-recommends -y \
    libcudnn7=7.6.0.64-1+cuda10.0  \
    libcudnn7-dev=7.6.0.64-1+cuda10.0

RUN apt-mark hold libcudnn7
RUN apt-mark hold cuda*

RUN apt-get update
RUN apt-get install -y nvinfer-runtime-trt-repo-ubuntu1804-5.0.2-ga-cuda10.0
RUN apt-get update
RUN apt-get install -y --no-install-recommends libnvinfer5=5.0.2-1+cuda10.0
RUN apt-get install -y --no-install-recommends libnvinfer-dev=5.0.2-1+cuda10.0 

# Install pip
RUN curl -O https://bootstrap.pypa.io/get-pip.py && \
    python3 get-pip.py && \
    rm get-pip.py

# Minimize image size 
RUN (apt-get autoremove -y; \
    apt-get autoclean -y)

# Set up work dir
WORKDIR /work

CMD /bin/bash
