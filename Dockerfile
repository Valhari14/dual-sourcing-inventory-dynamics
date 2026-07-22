FROM pytorch/pytorch:2.5.1-cuda11.8-cudnn9-runtime as base
RUN mkdir -p /opt/project/
WORKDIR /opt/project/
COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt

from base as app

COPY app/requirements.txt app/requirements.txt
RUN apt-get -o Acquire::Check-Valid-Until=false -o Acquire::Check-Date=false update && apt-get install  -y graphviz && pip install -r app/requirements.txt