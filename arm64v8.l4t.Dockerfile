ARG BASE_IMAGE=gcr.io/teknoir/nvidia/l4t-pytorch:r32.7.1-pth1.10-py3
FROM ${BASE_IMAGE}

RUN mkdir -p /usr/src/app
RUN git clone https://github.com/WongKinYiu/yolov7.git /usr/src/app
WORKDIR /usr/src/app

ENV PIP_BREAK_SYSTEM_PACKAGES 1
RUN python3 -m pip install --upgrade pip wheel
RUN python3 -m pip install --no-cache Pillow paho.mqtt numpy torch pandas requests torchvision opencv-python-headless==4.5.5.64 tqdm PyYAML matplot seaborn scipy
RUN python3 -m pip install --no-cache lap

ENV OMP_NUM_THREADS=1
ENV WANDB_MODE=disabled

ENV DEVICE=0

#ADD tracker tracker/
#ADD app.py .

CMD ["python3", "app.py"]
