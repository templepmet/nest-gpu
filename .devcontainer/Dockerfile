ARG VARIANT="11.6.2-devel-ubuntu20.04"
FROM nvcr.io/nvidia/cuda:${VARIANT}

ENV DEBIAN_FRONTEND=noninteractive
ENV DEBCONF_NOWARNINGS=yes

# Create non-root user
RUN apt-get update && apt-get install -y sudo
ARG USERNAME="dev"
ARG UID=1000
ARG GID=1000
RUN groupadd --gid $GID $USERNAME \
	&& useradd --uid $UID --gid $GID -m $USERNAME -s /bin/bash \
	&& echo "$USERNAME ALL=(ALL) NOPASSWD: ALL" > /etc/sudoers.d/$USERNAME \
	&& chmod 0440 /etc/sudoers.d/$USERNAME

RUN ln -sf /usr/share/zoneinfo/Asia/Tokyo /etc/localtime
RUN apt-get update \
	&& apt-get install -y --no-install-recommends \
	ca-certificates \
	git \
	curl \
	wget \
	patch \
	less \
	make \
	cmake \
	gcc \
	g++ \
	openmpi-bin \
	libopenmpi-dev \
	libomp-dev \
	openssl \
	libssl-dev \
	libbz2-dev \
	libreadline-dev \
	libsqlite3-dev \
	libffi-dev \
	zlib1g-dev \
	liblzma-dev \
	libtool \
	libtool-bin \
	locales-all
ENV CC=gcc
ENV CXX=g++
ENV CUDA_PATH=/usr/local/cuda

# Install pyenv
ENV PYTHON_VERSION="3.9.12"
ENV PYENV_ROOT="/home/${USERNAME}/.pyenv"
ENV PATH="${PYENV_ROOT}/shims:${PYENV_ROOT}/bin:${PATH}"
RUN git clone https://github.com/pyenv/pyenv.git ${PYENV_ROOT} \
	&& pyenv install -v ${PYTHON_VERSION} \
	&& pyenv global ${PYTHON_VERSION} \
	&& pyenv rehash

# Install pip package
RUN pip install matplotlib mpi4py

# Install nestgpu
ARG GIT_USER
ARG GIT_TOKEN
ENV NEST_GPU=/usr/local/nestgpu
ENV PYTHONPATH=${NEST_GPU}/lib/python3.9/site-packages:${PYTHONPATH}
USER ${USERNAME}
WORKDIR /home/${USERNAME}
RUN git clone https://${GIT_USER}:${GIT_TOKEN}@github.com/${GIT_USER}/nest-gpu.git/ installed_nest-gpu
WORKDIR /home/${USERNAME}/installed_nest-gpu
RUN ./install.sh
