FROM nvcr.io/nvidia/pytorch:24.01-py3

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    git \
    libglib2.0-0 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt/mast3r/dust3r
COPY preprocesamiento/models/mast3r /opt/mast3r
RUN pip install -r requirements.txt
RUN pip install -r requirements_optional.txt
RUN pip install opencv-python==4.8.0.74

WORKDIR /opt/mast3r
RUN pip install -r requirements.txt

COPY preprocesamiento/scripts/mast3r_reconstruct.py /opt/unwhere/mast3r_reconstruct.py

ENV PYTHONPATH=/opt/mast3r:/opt/mast3r/dust3r

ENTRYPOINT ["python", "/opt/unwhere/mast3r_reconstruct.py"]
