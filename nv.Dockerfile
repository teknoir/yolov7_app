
FROM gcr.io/teknoir/nvidia/pytorch:22.08-py3

RUN apt update && apt install --no-install-recommends -y zip htop screen libgl1-mesa-glx

RUN mkdir -p /usr/src/app
RUN git clone https://github.com/WongKinYiu/yolov7.git /usr/src/app
WORKDIR /usr/src/app
ENV PIP_BREAK_SYSTEM_PACKAGES 1
RUN pip install pip==21.1.1
RUN pip uninstall -y Pillow torchtext  # torch torchvision
RUN pip install --no-cache -r requirements.txt paho.mqtt albumentations wandb gsutil notebook Pillow>=9.1.0 \
    'opencv-python-headless==4.5.5.62' \
    --extra-index-url https://download.pytorch.org/whl/cu113

ENV OMP_NUM_THREADS=1
ENV WANDB_MODE=disabled

ADD tracker/ /usr/src/app/tracker/
COPY app.py .
COPY yolov7-tiny.pt .
COPY classes.names .

CMD ["python3", "app.py"]