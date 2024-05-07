ARG BASE_IMAGE=nvcr.io/nvidia/l4t-ml:r36.2.0-py3
FROM ${BASE_IMAGE}

RUN mkdir -p /usr/src/app
RUN git clone https://github.com/WongKinYiu/yolov7.git /usr/src/app
WORKDIR /usr/src/app

ENV PIP_BREAK_SYSTEM_PACKAGES 1
RUN python3 -m pip install --upgrade pip wheel
RUN python3 -m pip install --no-cache Pillow paho.mqtt==1.6.1 bson requests tqdm PyYAML matplot seaborn
RUN python3 -m pip install --no-cache lap

ENV OMP_NUM_THREADS=1
ENV WANDB_MODE=disabled

ENV DEVICE=0

ADD tracker tracker/
ADD app.py .


CMD ["python3", "app.py"]
