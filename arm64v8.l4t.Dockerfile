ARG BASE_IMAGE=gcr.io/teknoir/nvidia/l4t-pytorch:r32.7.1-pth1.10-py3
FROM ${BASE_IMAGE}

RUN mkdir -p /usr/src/app
RUN git clone https://github.com/WongKinYiu/yolov7.git /usr/src/app
WORKDIR /usr/src/app

ENV PIP_BREAK_SYSTEM_PACKAGES 1
RUN python3 -m pip install --upgrade pip wheel

#....PyTorch and TorchVision for Jetson Devices...
RUN pip install Cython
RUN pip install gdown
RUN gdown https://drive.google.com/uc?id=1e9FDGt2zGS5C5Pms7wzHYRb0HuupngK1
RUN pip install torch-1.13.0a0+git7c98e70-cp38-cp38-linux_aarch64.whl
RUN rm torch-1.13.0a0+git7c98e70-cp38-cp38-linux_aarch64.whl

RUN gdown https://drive.google.com/uc?id=19UbYsKHhKnyeJ12VPUwcSvoxJaX7jQZ2
RUN pip install torchvision-0.14.0a0+5ce4506-cp38-cp38-linux_aarch64.whl
RUN rm torchvision-0.14.0a0+5ce4506-cp38-cp38-linux_aarch64.whl
##################################################

RUN python3 -m pip install --no-cache Pillow paho.mqtt numpy pandas lap \
                                    requests opencv-python-headless==4.5.5.64 \
                                    tqdm PyYAML matplot seaborn scipy

ENV OMP_NUM_THREADS=1
ENV WANDB_MODE=disabled
ENV DEVICE=0

COPY app.py .
COPY yolov7-tiny.pt .
COPY classes.names .
ADD tracker/ /usr/src/app/tracker/ 

CMD ["python3", "app.py"]