FROM ubuntu:20.04

#ENV PYTHONUNBUFFERED 1
ENV DEBIAN_FRONTEND noniteractive
RUN apt-get update -y && apt-get upgrade -y
RUN apt-get -y install binutils vim git python3-pip wget unzip
#RUN apt-get update -y && apt-get upgrade -y
#RUN apt-get -y install libproj-dev gdal-bin libgdal-dev
RUN ln -sf /usr/bin/python3 /usr/bin/python
RUN ln -s /usr/bin/pip3 /usr/bin/pip
RUN pip install -U pip && hash -r pip

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
COPY requirements.txt /src/requirements.txt
RUN pip install -r /src/requirements.txt

RUN chmod -R a+w .

RUN jupyter notebook --generate-config

WORKDIR /app
# Install libraries for installing wgrib2
RUN apt-get update && apt-get install -y wget \
    build-essential \
    gfortran \
    zlib1g-dev

# Setting for libraries
ENV CC gcc
ENV FC gfortran

# Download wgrib2
RUN cd ~ \
    && wget ftp://ftp.cpc.ncep.noaa.gov/wd51we/wgrib2/wgrib2.tgz \
    && tar xvzf wgrib2.tgz

# Install wgrib2
RUN cd ~/grib2/ \
    && make \
    && cp wgrib2/wgrib2 /usr/local/bin/wgrib2

WORKDIR /src