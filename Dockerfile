FROM nvcr.io/nvidia/pytorch:21.08-py3

RUN rm -rf /opt/pytorch  # remove 1.2GB dir

RUN apt update && apt install --no-install-recommends -y zip htop screen libgl1-mesa-glx

RUN mkdir -p /usr/src/app
RUN git clone https://github.com/WongKinYiu/yolov7.git /usr/src/app
WORKDIR /usr/src/app

RUN python -m pip install --upgrade pip wheel
RUN pip uninstall -y Pillow torchtext  # torch torchvision
RUN pip install --no-cache -r requirements.txt albumentations wandb gsutil notebook Pillow>=9.1.0 \
    'opencv-python-headless==4.5.5.62' \
    --extra-index-url https://download.pytorch.org/whl/cu113

ENV OMP_NUM_THREADS=1
ENV WANDB_MODE=disabled

ADD app.py .
ADD app-keypoint.py .

CMD ["python3", "app.py"]
