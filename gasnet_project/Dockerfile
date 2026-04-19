FROM pytorch/pytorch:2.3.1-cuda12.1-cudnn8-runtime

LABEL org.opencontainers.image.title="gasnet-h100" \
    org.opencontainers.image.description="GASNet training image for NVIDIA H100 (CUDA 12.1+)"

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility \
    TORCHINDUCTOR_CACHE_DIR=/tmp/torchinductor

WORKDIR /workspace/gasnet_project

RUN mkdir -p ${TORCHINDUCTOR_CACHE_DIR}

COPY requirements.txt ./
RUN pip install -r requirements.txt

COPY train.py dataset.py utils.py evaluation.py ./

# Dataset is mounted at runtime, not copied into image.
# Example run:
# docker run --gpus all --rm -it \
#   -v /absolute/path/to/VRU_parent:/data \
#   -v /absolute/path/to/output:/workspace/output \
#   gasnet-h100:latest \
#   python train.py --data-root /data --epochs 60 --batch-size 512

# Build:
# docker build -t gasnet-h100:latest ./gasnet_project

CMD ["python", "train.py", "--help"]
