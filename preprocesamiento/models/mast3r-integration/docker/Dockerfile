# syntax=docker/dockerfile:1.4
# ══════════════════════════════════════════════════════════════════════════════
# Single-stage con BuildKit cache mounts
#
# Beneficios vs Dockerfile original:
#   • --mount=type=cache  → apt y pip NO se bajan de nuevo en rebuilds
#     (los cachés nunca se graban como capas → imagen más limpia)
#   • Requirements copiados ANTES del código fuente → la capa de pip solo
#     se invalida cuando cambia requirements.txt, no cuando cambia el código
#   • Parches Speedy MASt3R aplicados y eliminados tras el patch
#     (blocks.py FlashAttention + fast_nn.py FastNN-Lite/HybridCast)
#   • Artefactos temporales de build limpiados con rm -rf al final
# ══════════════════════════════════════════════════════════════════════════════
FROM nvcr.io/nvidia/pytorch:24.01-py3

ARG DEBIAN_FRONTEND=noninteractive

# ── Sistema ──────────────────────────────────────────────────────────────────
# Cache mount: apt nunca se descarga de nuevo en rebuilds; no añade capa
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    apt-get update && apt-get install -y --no-install-recommends \
        git curl libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# ── uv ───────────────────────────────────────────────────────────────────────
RUN curl -LsSf https://astral.sh/uv/install.sh | UV_UNMANAGED_INSTALL="/usr/local/bin" sh

# ── Dependencias dust3r (capa cacheada antes del COPY de fuentes) ─────────
# Clonar solo para los requirements; --filter=blob:none evita descargar LFS
RUN git clone --depth=1 --filter=blob:none \
        https://github.com/naver/dust3r /tmp/dust3r_reqs

RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --system \
        -r /tmp/dust3r_reqs/requirements.txt \
        -r /tmp/dust3r_reqs/requirements_optional.txt \
        "opencv-python==4.8.0.74"

# ── Dependencias MASt3R (COPY requirements solo, no todo el repo) ─────────
COPY preprocesamiento/models/mast3r/requirements.txt /tmp/mast3r-reqs.txt
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --system -r /tmp/mast3r-reqs.txt

# ── faiss-cpu (retrieval scene graph) ────────────────────────────────────
# faiss-gpu crashea con cuBLAS en CUDA 12.3
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --system faiss-cpu

# ── flash-attn (Speedy MASt3R — FlashAttention en encoder self-attention) ─
# Necesita compilar contra PyTorch/CUDA del sistema → --no-build-isolation
# --system install evita el problema de --target que rompe el build de CUDA
# MAX_JOBS=4 evita OOM durante la compilación paralela de CUDA kernels
RUN --mount=type=cache,target=/root/.cache/pip \
    MAX_JOBS=4 pip install flash-attn --no-build-isolation

# ── cython + asmk (ASMK retrieval index) ─────────────────────────────────
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --system cython
RUN git clone --depth=1 https://github.com/jenicek/asmk /opt/asmk && \
    cd /opt/asmk/cython && python -m cython *.pyx && \
    cd /opt/asmk && uv pip install --system -e .

# ── Pin numpy<2 (al final — compatibilidad ABI PyTorch 2.2) ─────────────
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --system "numpy<2"

# ── Código fuente MASt3R + dust3r submodule ───────────────────────────────
# Esta capa se invalida solo cuando cambia el código, no las deps anteriores
COPY preprocesamiento/models/mast3r /opt/mast3r

RUN if [ ! -f /opt/mast3r/dust3r/requirements.txt ]; then \
        rm -rf /opt/mast3r/dust3r && \
        git clone --depth=1 --recursive https://github.com/naver/dust3r /opt/mast3r/dust3r; \
    fi

# ── Parches Speedy MASt3R ─────────────────────────────────────────────────
# Aplica solo los 2 ficheros relevantes para nuestro pipeline:
#   blocks.py  → FlashAttention en encoder self-attention (53ms→28ms/par)
#   fast_nn.py → FastNN-Lite + HybridCast en matching    (116ms→18ms/par)
# El repo se clona y elimina en el mismo RUN → no queda en la imagen final
RUN git clone --depth=1 \
        https://github.com/ASU-ESIC-FAN-Lab/speedy_mast3r /tmp/speedy && \
    cp /tmp/speedy/dust3r/croco/models/blocks.py \
       /opt/mast3r/dust3r/croco/models/blocks.py && \
    cp /tmp/speedy/mast3r/fast_nn.py \
       /opt/mast3r/mast3r/fast_nn.py && \
    rm -rf /tmp/speedy /tmp/dust3r_reqs /tmp/mast3r-reqs.txt

# Compatibilidad flash-attn: flash_attn_with_kvcache se añadió en ≥2.1.0;
# el entorno tiene 2.0.4.  Solo flash_attn_func es usada en Attention.forward
# (encoder self-attention) → hacemos el import de los otros dos opcional.
RUN python -c "\
p='/opt/mast3r/dust3r/croco/models/blocks.py'; \
t=open(p).read(); \
old='from flash_attn import flash_attn_qkvpacked_func, flash_attn_func,flash_attn_with_kvcache'; \
new='from flash_attn import flash_attn_func\ntry:\n    from flash_attn import flash_attn_qkvpacked_func, flash_attn_with_kvcache\nexcept ImportError:\n    flash_attn_qkvpacked_func = None; flash_attn_with_kvcache = None'; \
assert old in t, 'import not found'; \
open(p,'w').write(t.replace(old,new)); \
print('blocks.py patched OK') \
"

# ── Compilar kernels CUDA RoPE (embeddings posicionales) ─────────────────
WORKDIR /opt/mast3r/dust3r/croco/models/curope/
RUN python setup.py build_ext --inplace

# ── Script de integración UniWhere ────────────────────────────────────────
COPY preprocesamiento/models/mast3r-integration/mast3r_sfm.py /opt/unwhere/mast3r_sfm.py

ENV PYTHONPATH=/opt/mast3r:/opt/mast3r/dust3r

ENTRYPOINT ["python", "/opt/unwhere/mast3r_sfm.py"]
