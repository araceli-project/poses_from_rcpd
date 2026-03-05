FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

ENV DEBIAN_FRONTEND noninteractive

# install essentials
RUN apt-get update -y && apt-get install -y \
  software-properties-common \
  build-essential \
  libblas-dev \
  libhdf5-serial-dev \
  git \
  python3-venv \
  zsh tmux htop vim curl wget unzip ranger fish

RUN curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | bash
RUN apt-get install git-lfs

# install some niceties
RUN pip3 install -U pip
RUN ln -s /usr/bin/python3 /usr/bin/python

# make terminals look pretty (setting a reasonable colour setting)
RUN touch /usr/share/locale/locale.alias
RUN apt-get -y install locales
RUN sed -i -e 's/# en_US.UTF-8 UTF-8/en_US.UTF-8 UTF-8/' /etc/locale.gen && \
  dpkg-reconfigure --frontend=noninteractive locales && \
  update-locale LANG=en_US.UTF-8
ENV LANG en_US.UTF-8
ENV TERM xterm-256color

# extra stuff that I needed along the way
RUN apt-get update -y && apt-get install -y graphviz pkg-config --fix-missing
RUN apt-get update -y && apt-get install -y libcairo2 libcairo2-dev --fix-missing
RUN apt-get autoremove -y && apt-get autoclean -y

# change workdir and add some files
WORKDIR /src

# install our beloved requirements
RUN pip install -U pip

RUN pip install Cython
RUN pip install --no-build-isolation -U git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI
RUN pip install git+https://github.com/facebookresearch/segment-anything.git

# install MMCV
RUN apt-get update && apt-get install -y ffmpeg libsm6 libxext6 git ninja-build libglib2.0-0 libsm6 libxrender-dev libxext6 \
  && apt-get clean \
  && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir --upgrade pip wheel setuptools
RUN pip install --no-cache-dir mmcv==2.0.0 -f https://download.openmmlab.com/mmcv/dist/cu117/torch2.0/index.html
RUN pip install openmim
RUN mim install mmdet==2.25.3

WORKDIR /
