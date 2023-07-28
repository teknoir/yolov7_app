ARG BASE_IMAGE=gcr.io/teknoir/yolov7:l4tr32.7.1
FROM ${BASE_IMAGE}

ARG MODEL_NAME=yolov7-tiny
ENV MODEL_NAME=$MODEL_NAME

# ARG IMG_SIZE=640
# ENV IMG_SIZE=$IMG_SIZE

# ARG WEIGHTS_FILE=yolov7-tiny.pt
# ENV WEIGHTS_FILE=$WEIGHTS_FILE
# ENV WEIGHTS=/usr/src/app/yolov7-tiny.pt
# ADD $WEIGHTS_FILE $WEIGHTS

# ARG CLASS_NAMES_FILE=classes.names
# ENV CLASS_NAMES_FILE=$CLASS_NAMES_FILE
# ENV CLASS_NAMES=/usr/src/app/classes.names
# ADD $CLASS_NAMES_FILE $CLASS_NAMES

ADD classes.names /usr/src/app/classes.names
ADD yolov7-tiny.pt /usr/src/app/yolov7-tiny.pt
ADD app.py /usr/src/app/app.py
ADD tracker/ /usr/src/tracker/