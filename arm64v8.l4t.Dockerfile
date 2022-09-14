ARG BASE_IMAGE=nvcr.io/nvidia/l4t-pytorch:r32.7.1-pth1.10-py3
FROM ${BASE_IMAGE} as base

RUN python3 -m pip install --upgrade pip wheel
RUN python3 -m pip install --no-cache Pillow paho.mqtt numpy torch pandas requests torchvision opencv-python-headless==4.5.5.64 tqdm PyYAML matplot seaborn scipy

RUN mkdir -p /usr/src/app
WORKDIR /usr/src/app

ADD . /usr/src/app

ENV OMP_NUM_THREADS=1
ENV WANDB_MODE=disabled
ENV DEVICE=0

CMD ["python3", "app.py"]
