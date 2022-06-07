# See here for image contents: https://github.com/microsoft/vscode-dev-containers/tree/v0.209.6/containers/cpp/.devcontainer/base.Dockerfile

# [Choice] Debian / Ubuntu version (use Debian 11/9, Ubuntu 18.04/21.04 on local arm64/Apple Silicon): debian-11, debian-10, debian-9, ubuntu-21.04, ubuntu-20.04, ubuntu-18.04
ARG VARIANT="bullseye"
FROM mcr.microsoft.com/vscode/devcontainers/cpp:0-${VARIANT}

# [Optional] Uncomment this section to install additional packages.
# RUN apt-get update && export DEBIAN_FRONTEND=noninteractive \
#     && apt-get -y install --no-install-recommends <your-package-list-here>
ENV DEBIAN_FRONTEND=noninteractive

ENV PYTHON_VERSION="3.9.5"
ENV PYENV_ROOT="/.pyenv"
ENV PATH="${PYENV_ROOT}/shims:${PYENV_ROOT}/bin:${PATH}"
ENV PWD="/workspaces/nest-gpu"
# ENV QULACS_OSAKA_PACKAGE="qulacs_osaka"
# ENV QULACS_OSAKA_VERSION="0.3.1"
# Add build artifact to PYTHONPATH and python can find qulacs.
# Egg file name might vary depending on qulacs and python version.
# ENV PYTHONPATH="${PWD}/dist/${QULACS_OSAKA_PACKAGE}-${QULACS_OSAKA_VERSION}-cp39-cp39-linux_x86_64.whl:${PYTHONPATH}"
# ENV PYTHONPATH="${PWD}/build/lib.linux-x86_64-3.9:${PYTHONPATH}"

RUN apt-get update \
  && apt-get install -y --no-install-recommends \
  git

# Install pyenv
RUN git clone https://github.com/pyenv/pyenv.git ${PYENV_ROOT} \
  && pyenv install ${PYTHON_VERSION} \
  && pyenv global ${PYTHON_VERSION} \
  && pyenv rehash \
  && python -m pip install --upgrade pip \
  && pip install matplotlib mpi4py \
  && chmod -R 777 /.pyenv
