FROM nvcr.io/nvidia/pytorch:24.01-py3

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    git \
    libglib2.0-0 \
    curl \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Instalar uv (gestor de paquetes más rápido que pip para builds repetidos)
# Fijar versión para builds reproducibles
RUN curl -LsSf https://astral.sh/uv/install.sh | UV_UNMANAGED_INSTALL="/usr/local/bin" sh

# Copiar MASt3R y asegurar dust3r submodule
COPY preprocesamiento/models/mast3r /opt/mast3r
RUN if [ ! -f /opt/mast3r/dust3r/requirements.txt ]; then \
        rm -rf /opt/mast3r/dust3r && \
        git clone --recursive https://github.com/naver/dust3r /opt/mast3r/dust3r; \
    fi

# Instalar dependencias con uv --system
WORKDIR /opt/mast3r/dust3r
RUN uv pip install --system -r requirements.txt
RUN uv pip install --system -r requirements_optional.txt
RUN uv pip install --system "opencv-python==4.8.0.74"

# Pin numpy<2 DESPUÉS de todos los installs — compatibilidad con PyTorch 2.2 del base image Jan 2024
RUN uv pip install --system "numpy<2"

# Compilar kernels CUDA RoPE (embeddings posicionales, acelera inferencia)
WORKDIR /opt/mast3r/dust3r/croco/models/curope/
RUN python setup.py build_ext --inplace

WORKDIR /opt/mast3r
RUN uv pip install --system -r requirements.txt

# Copiar script principal desde mast3r-integration/ (junto al Dockerfile)
COPY preprocesamiento/models/mast3r-integration/mast3r_sfm.py /opt/unwhere/mast3r_sfm.py

ENV PYTHONPATH=/opt/mast3r:/opt/mast3r/dust3r

ENTRYPOINT ["python", "/opt/unwhere/mast3r_sfm.py"]
