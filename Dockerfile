FROM pytorch/pytorch:2.1.2-cuda11.8-cudnn8-devel

ENV DEBIAN_FRONTEND=noninteractive PIP_PREFER_BINARY=1

RUN apt -y update && apt -y install git vim less python3-pip wget curl libgl1-mesa-dev libglib2.0-0

WORKDIR /app
RUN pip3 install transformers \
        Pillow
