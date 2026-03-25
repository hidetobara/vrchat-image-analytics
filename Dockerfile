FROM pytorch/pytorch:2.7.1-cuda12.8-cudnn9-devel

ENV DEBIAN_FRONTEND=noninteractive
ENV PIP_PREFER_BINARY=1
ENV TOKENIZERS_PARALLELISM=false

RUN apt -y update && apt -y install git vim less python3-pip wget curl libgl1-mesa-dev libglib2.0-0

WORKDIR /app
RUN pip3 install transformers \
        Pillow \
        requests \
        tqdm

ENV PYTHONPATH="/app/src"
