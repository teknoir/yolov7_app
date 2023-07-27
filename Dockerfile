FROM python:3.9

RUN apt-get update && \
    apt-get install -y --no-install-recommends --allow-downgrades \
    build-essential \
    git \
    gcc \
    wget \
    libgl1-mesa-glx \
    libglib2.0-0 \
    python3-dev \
    python3-pip

ENV PIP_BREAK_SYSTEM_PACKAGES=1
RUN pip install pip==21.1.1

RUN mkdir -p /usr/src/app
RUN git clone https://github.com/WongKinYiu/yolov7.git /usr/src/app
WORKDIR /usr/src/app

RUN pip install --no-cache -r requirements.txt paho.mqtt albumentations wandb gsutil notebook Pillow>=9.1.0 \
    'opencv-python-headless==4.5.5.62'  \ 
    torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cpu
RUN pip install lap 

ENV OMP_NUM_THREADS=1
ENV WANDB_MODE=disabled
ENV DEVICE=0

COPY tracker tracker/
COPY app.py .
COPY yolov7-tiny.onnx .
COPY classes.names .
CMD ["python3", "app.py"]
