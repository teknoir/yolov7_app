FROM debian:bullseye
#  
RUN apt-get update && \
    apt-get install build-essential -y --no-install-recommends --allow-downgrades \
    git wget curl tar unzip zip htop screen \
    gcc libgl1-mesa-glx libglib2.0-0 \
    python3 \
    python3-dev \
    python3-pip \
    build-essential

ENV PIP_BREAK_SYSTEM_PACKAGES 1
RUN pip --version
RUN mkdir -p /usr/src/app
RUN git clone https://github.com/WongKinYiu/yolov7.git /usr/src/app
WORKDIR /usr/src/app

# RUN python3 -m pip install --upgrade pip wheel
RUN python3 -m pip install --no-cache -r requirements.txt paho.mqtt \
    Pillow>=9.1.0 opencv-python-headless==4.5.5.62 torch torchvision torchaudio \
    --extra-index-url https://download.pytorch.org/whl/cpu 

ENV OMP_NUM_THREADS=1
ENV WANDB_MODE=disabled

ADD app.py .

CMD ["python3", "app.py"]
