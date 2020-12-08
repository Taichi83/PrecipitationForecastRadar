#FROM nvidia/cuda:11.1-devel-ubuntu20.04
FROM nvidia/cuda:11.1-devel-ubuntu18.04

ENV LANG C.UTF-8
RUN APT_INSTALL="apt-get install -y --no-install-recommends" && \
    PIP_INSTALL="python -m pip --no-cache-dir install --upgrade" && \
    GIT_CLONE="git clone --depth 10" && \

    rm -rf /var/lib/apt/lists/* \
           /etc/apt/sources.list.d/cuda.list \
           /etc/apt/sources.list.d/nvidia-ml.list && \

    apt-get update && \

# ==================================================================
# tools
# ------------------------------------------------------------------

    DEBIAN_FRONTEND=noninteractive $APT_INSTALL \
        build-essential \
        apt-utils \
        ca-certificates \
        wget \
        git \
        vim \
        libssl-dev \
        curl \
        unzip \
        unrar \
        && \

    $GIT_CLONE https://github.com/Kitware/CMake ~/cmake && \
    cd ~/cmake && \
    ./bootstrap && \
    make -j"$(nproc)" install && \

# ==================================================================
# python
# ------------------------------------------------------------------

    DEBIAN_FRONTEND=noninteractive $APT_INSTALL \
        software-properties-common \
        && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update && \
    DEBIAN_FRONTEND=noninteractive $APT_INSTALL \
        python3.6 \
        python3.6-dev \
        python3-distutils-extra \
        && \
    wget -O ~/get-pip.py \
        https://bootstrap.pypa.io/get-pip.py && \
    python3.6 ~/get-pip.py && \
    ln -s /usr/bin/python3.6 /usr/local/bin/python3 && \
    ln -s /usr/bin/python3.6 /usr/local/bin/python && \
    $PIP_INSTALL \
        setuptools \
        && \
    $PIP_INSTALL \
        numpy \
        scipy \
        pandas \
        cloudpickle \
        scikit-image>=0.14.2 \
        scikit-learn \
        matplotlib \
        Cython \
        tqdm \
        && \

# ==================================================================
# pytorch
# ------------------------------------------------------------------

    $PIP_INSTALL \
        future \
        numpy \
        protobuf \
        enum34 \
        pyyaml \
        typing \
        && \
    $PIP_INSTALL \
#        --pre torch torchvision -f \
#        https://download.pytorch.org/whl/nightly/cu101/torch_nightly.html \
        --pre torch torchvision torchaudio -f \
        https://download.pytorch.org/whl/nightly/cu110/torch_nightly.html \
        && \

# ==================================================================
# config & cleanup
# ------------------------------------------------------------------

    ldconfig && \
    apt-get clean && \
    apt-get autoremove && \
    rm -rf /var/lib/apt/lists/* /tmp/* ~/*

ENV PYTHONUNBUFFERED 1
#ENV DEBIAN_FRONTEND noniteractive
RUN apt-get update -y && apt-get upgrade -y
RUN apt-get -y install binutils vim git python3-pip wget unzip
#RUN apt-get update -y && apt-get upgrade -y
#RUN apt-get -y install libproj-dev gdal-bin libgdal-dev
RUN ln -sf /usr/bin/python3 /usr/bin/python
RUN ln -s /usr/bin/pip3 /usr/bin/pip
RUN pip install -U pip && hash -r pip

# For rasterio
RUN apt-get install software-properties-common -y
RUN add-apt-repository ppa:ubuntugis/ppa
RUN apt-get update -y
RUN apt-get -y install python-numpy gdal-bin libgdal-dev
RUN pip install rasterio

USER root
RUN mkdir /src
WORKDIR /src

# install nodejs and npm for plotly on jupyterlab
ENV NODEJS_VERSION v12
RUN apt-get install -y curl
ENV PATH=/root/.nodebrew/current/bin:$PATH
RUN curl -L git.io/nodebrew | perl - setup && \
    nodebrew install-binary ${NODEJS_VERSION} && \
    nodebrew use ${NODEJS_VERSION}

# pip
RUN pip install -U pip
COPY requirements-gpu.txt /src/requirements-gpu.txt
RUN pip install -r /src/requirements-gpu.txt

# jax
# Install JAX

##RUN ln -s /usr/local/cuda /usr/local/cuda-11.1
ENV XLA_FLAGS --xla_gpu_cuda_data_dir=/usr/local/cuda/bin
RUN pip install --upgrade jaxlib==0.1.57+cuda111 -f https://storage.googleapis.com/jax-releases/jax_releases.html
RUN pip install --upgrade jax
RUN mkdir /app
WORKDIR /app
RUN git clone https://github.com/pyro-ppl/numpyro.git
WORKDIR /app/numpyro
RUN pip install -e .[dev]
#RUN pip install numpyro


WORKDIR /src
RUN chmod -R a+w .

RUN jupyter notebook --generate-config
ENV CURL_CA_BUNDLE /etc/ssl/certs/ca-certificates.crt
#WORKDIR /app
## Install libraries for installing wgrib2
#RUN apt-get update && apt-get install -y wget \
#    build-essential \
#    gfortran \
#    zlib1g-dev
#
## Setting for libraries
#ENV CC gcc
#ENV FC gfortran
#
## Download wgrib2
#RUN cd ~ \
#    && wget ftp://ftp.cpc.ncep.noaa.gov/wd51we/wgrib2/wgrib2.tgz \
#    && tar xvzf wgrib2.tgz
#
## Install wgrib2
#RUN cd ~/grib2/ \
#    && make \
#    && cp wgrib2/wgrib2 /usr/local/bin/wgrib2

WORKDIR /src