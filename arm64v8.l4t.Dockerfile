ARG BASE_IMAGE=gcr.io/teknoir/nvidia/l4t-pytorch:r32.7.1-pth1.10-py3
FROM ${BASE_IMAGE}

RUN mkdir -p /usr/src/app
RUN git clone https://github.com/WongKinYiu/yolov7.git /usr/src/app
WORKDIR /usr/src/app

ENV PIP_BREAK_SYSTEM_PACKAGES 1
RUN python3 -m pip install --upgrade pip wheel

#....PyTorch and TorchVision for Jetson Devices...
RUN apt-get install -y --no-install-recommends libjpeg-dev libopenblas-dev libopenmpi-dev \
                                                libomp-dev libavcodec-dev libavformat-dev libswscale-dev
RUN pip install scikit-build setuptools==58.3.0 Cython gdown future

RUN gdown https://drive.google.com/uc?id=1AQQuBS9skNk1mgZXMp0FmTIwjuxc81WY
RUN pip install torch-1.11.0a0+gitbc2c6ed-cp38-cp38-linux_aarch64.whl
RUN rm torch-1.11.0a0+gitbc2c6ed-cp38-cp38-linux_aarch64.whl    

RUN gdown https://drive.google.com/uc?id=1BaBhpAizP33SV_34-l3es9MOEFhhS1i2
RUN pip install torchvision-0.12.0a0+9b5a3fe-cp38-cp38-linux_aarch64.whl
RUN rm torchvision-0.12.0a0+9b5a3fe-cp38-cp38-linux_aarch64.whl
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