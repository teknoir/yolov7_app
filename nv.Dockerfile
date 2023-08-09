FROM gcr.io/teknoir/nvidia/pytorch:22.08-py3

RUN apt update && apt install --no-install-recommends -y zip htop screen libgl1-mesa-glx

RUN mkdir -p /usr/src/app
RUN git clone https://github.com/WongKinYiu/yolov7.git /usr/src/app

WORKDIR /usr/src/app

ENV PIP_BREAK_SYSTEM_PACKAGES 1
RUN python -m pip install --upgrade pip wheel
RUN pip uninstall -y Pillow torchtext  # torch torchvision
RUN pip install --no-cache -r requirements.txt paho.mqtt Pillow>=9.1.0 \
    opencv-python-headless==4.5.5.62 --extra-index-url https://download.pytorch.org/whl/cu113
# NOT USED:  albumentations wandb gsutil notebook

# OBJECT TRACKING DEPENDENCIES
RUN python3 -m pip install numpy scipy lap

ENV OMP_NUM_THREADS=1
ENV WANDB_MODE=disabled
ADD tracker tracker/
ADD app.py .

#RUN wget https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7.pt

CMD ["python3", "app.py"]