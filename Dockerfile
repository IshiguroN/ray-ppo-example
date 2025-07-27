FROM nvidia/cuda:12.8.1-cudnn-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

# Install language
RUN apt-get update && apt-get install -y --no-install-recommends \
  locales \
  && locale-gen ja_JP.UTF-8 \
  && update-locale LC_ALL=ja_JP.UTF-8 LANG=ja_JP.UTF-8 \
  && rm -rf /var/lib/apt/lists/*
ENV LANG=ja_JP.UTF-8

# Install timezone
RUN ln -fs /usr/share/zoneinfo/UTC /etc/localtime \
  && export DEBIAN_FRONTEND=noninteractive \
  && apt-get update \
  && apt-get install -y --no-install-recommends tzdata \
  && dpkg-reconfigure --frontend noninteractive tzdata \
  && rm -rf /var/lib/apt/lists/*


# Install system dependencies
RUN apt-get update && apt-get install -y \
    cmake \
    build-essential \
    libgl1-mesa-glx \
    libgl1-mesa-dri \
    libx11-xcb1 \
    python3-pip \
    python3-dev \
    python3-opengl \
    swig \
    xvfb \
    x11-apps \
    libglib2.0-0 \
    libglib2.0-dev \
    tk-dev \
    python3.10-tk \
    vim \
    && rm -rf /var/lib/apt/lists/* \
    && ldconfig \
    && ls -l /usr/lib/x86_64-linux-gnu/libGL.so* \
    && ls -l /usr/lib/x86_64-linux-gnu/libgthread-2.0.so*

# Install gymnasium and additional dependencies
RUN pip install --upgrade pip
# RUN pip install torch torchrl gymnasium[classic-control,box2d] vmas tqdm pettingzoo[mpe]==1.24.3 tensorboard matplotlib
# https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
RUN pip install "ray[rllib]" torch  gymnasium[classic-control,box2d,atari,accept-rom-license,mujoco] tqdm tensorboard matplotlib pandas seaborn 

WORKDIR /rllib_ws
CMD ["/bin/bash"]
