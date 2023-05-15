FROM debian:stable
#  
RUN apt-get update && \
    apt-get install build-essential -y --no-install-recommends --allow-downgrades \
    git wget curl tar unzip zip htop screen \
    gcc libgl1-mesa-glx libglib2.0-0 \
    python3 \
    python3-dev \
    python3-pip

RUN pip --version
RUN mkdir -p /usr/src/app
RUN git clone https://github.com/WongKinYiu/yolov7.git /usr/src/app
WORKDIR /usr/src/app

RUN python3 -m pip install --upgrade pip wheel
RUN python3 -m pip install --no-cache -r requirements.txt paho.mqtt albumentations wandb gsutil notebook Pillow>=9.1.0 \
    'opencv-python-headless==4.5.5.62'  \ 
    setuptools==58.4 torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cpu
RUN python3 -m pip install numpy lap

ENV OMP_NUM_THREADS=1
ENV WANDB_MODE=disabled

RUN wget https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7-tiny.pt
ADD app.py .
ADD tracker ./tracker
ADD coco.names .
CMD ["python3", "app.py"]