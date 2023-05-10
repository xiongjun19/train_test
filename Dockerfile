
ARG DOCKER_VERSION=22.09
ARG BASE_IMAGE=nvcr.io/nvidia/pytorch:${DOCKER_VERSION}-py3
FROM ${BASE_IMAGE}

RUN apt-get update && \
    apt-get install -y --no-install-recommends bc git-lfs&& \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN apt-get update && apt-get install -y sysstat
RUN python -m pip install --upgrade pip
ADD requirements.txt 
RUN pip install -r requirements.txt
