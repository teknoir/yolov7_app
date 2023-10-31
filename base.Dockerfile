# As L4T and cross compilation is a tough nut, only the base image have to be built on the device
# If we only add files in the last layers of the image, it can be done without cross compilation or directly on the device
# I.e. models, weights, settings and python files can be added here, as long as there is nothing executed (no adding of deps etc.)

ARG BASE_IMAGE=gcr.io/teknoir/yolov7:l4tr32.7.1
FROM ${BASE_IMAGE}

ADD app.py /usr/src/app/app.py
ADD tracker/ /usr/src/app/tracker/