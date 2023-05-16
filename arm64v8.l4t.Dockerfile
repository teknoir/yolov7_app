ARG BASE_IMAGE=gcr.io/teknoir/nvidia/l4t-pytorch:r32.7.1-pth1.10-py3
FROM ${BASE_IMAGE}

RUN mkdir -p /usr/src/app
RUN git clone https://github.com/WongKinYiu/yolov7.git /usr/src/app
WORKDIR /usr/src/app

RUN python3 -m pip install --upgrade pip wheel
RUN python3 -m pip install --no-cache Pillow paho.mqtt numpy torch pandas requests torchvision opencv-python-headless==4.5.5.64 tqdm PyYAML matplot seaborn scipy

ENV OMP_NUM_THREADS=1
ENV WANDB_MODE=disabled
ENV DEVICE=0

ADD app.py .
ADD object.names .
RUN wget https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7-tiny.pt

CMD ["python3", "app.py"]
