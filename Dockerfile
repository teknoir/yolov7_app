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

ENV PIP_BREAK_SYSTEM_PACKAGES 1
RUN pip install pip==21.1.1

RUN mkdir -p /usr/src/app
RUN git clone https://github.com/WongKinYiu/yolov7.git /usr/src/app
WORKDIR /usr/src/app

RUN python3 -m pip install --no-cache -r requirements.txt paho.mqtt albumentations wandb gsutil notebook Pillow>=9.1.0 \
    'opencv-python-headless==4.5.5.62'  \ 
    torch lap torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cpu
RUN pip install numpy==1.20.3
ENV OMP_NUM_THREADS=1
ENV WANDB_MODE=disabled
ENV DEVICE=0

ADD tracker/ /usr/src/app/tracker/
COPY app.py .
COPY yolov7-tiny.pt .
COPY classes.names .
CMD ["python3", "app.py"]
