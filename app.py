#!/usr/bin/env python

import os
import sys
import json
import time
import base64
import gc
import logging
import torch
import torch.backends.cudnn as cudnn
import numpy as np
import paho.mqtt.client as mqtt
from io import BytesIO
from PIL import Image

# these imports draw from yolov7, which is cloned when the dockerfiles are built
from utils.datasets import letterbox
from utils.torch_utils import select_device
from utils.general import non_max_suppression, check_img_size, scale_coords
from models.experimental import attempt_load

import warnings
warnings.filterwarnings('ignore')


APP_NAME = os.getenv('APP_NAME', 'yolov7')

args = {
    'MQTT_IN_0': os.getenv("MQTT_IN_0", f"{APP_NAME}/images"),
    'MQTT_OUT_0': os.getenv("MQTT_OUT_0", f"{APP_NAME}/detections"),
    'MQTT_OUT_1': os.getenv("MQTT_OUT_1", f"{APP_NAME}/movements"),
    'MQTT_VERSION': os.getenv("MQTT_VERSION", '3'),
    'MQTT_TRANSPORT': os.getenv("MQTT_TRANSPORT", 'tcp'),
    'MQTT_SERVICE_HOST': os.getenv('MQTT_SERVICE_HOST', '127.0.0.1'),
    'MQTT_SERVICE_PORT': int(os.getenv('MQTT_SERVICE_PORT', '1883')),
    'DEVICE': os.getenv("DEVICE", '0'),
    'WEIGHTS': str(os.getenv("WEIGHTS", "model.pt")),
    'AGNOSTIC_NMS': bool(os.getenv("AGNOSTIC_NMS", "False")),
    'IMG_SIZE': int(os.getenv("IMG_SIZE", "640")),
    'CLASS_NAMES': os.getenv("CLASS_NAMES", "obj.names"),
    'IOU_THRESHOLD': float(os.getenv("IOU_THRESHOLD", "0.45")),
    'CONF_THRESHOLD': float(os.getenv("CONF_THRESHOLD", "0.25")),
    'AUGMENTED_INFERENCE': bool(os.getenv("AUGMENTED_INFERENCE", "False")),
    'CLASSES_TO_DETECT': str(os.getenv("CLASSES_TO_DETECT", "person,bicycle,car,motorbike,truck")),
}

logger = logging.getLogger(APP_NAME)
ch = logging.StreamHandler(sys.stdout)
ch.setLevel(logging.INFO)
formatter = logging.Formatter(
    '[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)
logger.setLevel(logging.INFO)

logger.info("TΞꓘN01R")
logger.info("TΞꓘN01R")
logger.info("TΞꓘN01R")

logger.info(json.dumps(args))


def error_str(rc):
    return '{}: {}'.format(rc, mqtt.error_string(rc))


def on_connect_v3(client, _userdata, _flags, rc):
    logger.info('Connected to MQTT broker {}'.format(error_str(rc)))
    if rc == 0:
        client.subscribe(args['MQTT_IN_0'], qos=0)


def on_connect_v5(client, _userdata, _flags, rc, _props):
    logger.info('Connected to MQTT broker {}'.format(error_str(rc)))
    if rc == 0:
        client.subscribe(args['MQTT_IN_0'], qos=0)


def base64_encode(ndarray_image):
    buff = BytesIO()
    Image.fromarray(ndarray_image).save(buff, format='JPEG')
    string_encoded = base64.b64encode(buff.getvalue()).decode("utf-8")
    return f"data:image/jpeg;base64,{string_encoded}"


class NumpyEncoder(json.JSONEncoder):

    def default(self, obj,):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.int32):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.array):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


if args["CLASS_NAMES"] == "":
    logger.error("You must specify 'CLASS_NAMES'")
    sys.exit(1)

class_names = []
with open(args["CLASS_NAMES"], "r", encoding='utf-8') as names_file:
    for line in names_file:
        if line != "" and line != "\n":
            class_names.append(line.strip())
args["CLASS_NAMES"] = class_names

if args["CLASSES_TO_DETECT"] == "":
    args["CLASSES_TO_DETECT"] = list(range(len(class_names)))
else:
    class_ids = []
    for class_name in [s.strip() for s in args["CLASSES_TO_DETECT"].split(",")]:
        index = args["CLASS_NAMES"].index(class_name)
        class_ids.append(index)
        args["CLASSES_TO_DETECT"] = class_ids

logger.info("Loading YOLOv7 Model")
device = select_device(args["DEVICE"])
half = device.type != 'cpu'  # half precision only supported on CUDA
model = attempt_load(args["WEIGHTS"], map_location=device)
stride = int(model.stride.max())
imgsz = check_img_size(args["IMG_SIZE"], s=stride)
if isinstance(imgsz, (list, tuple)):
    assert len(imgsz) == 2
    "height and width of image has to be specified"
    imgsz[0] = check_img_size(imgsz[0], s=stride)
    imgsz[1] = check_img_size(imgsz[1], s=stride)
else:
    imgsz = check_img_size(imgsz, s=stride)
names = model.module.names if hasattr(model, 'module') else model.names
if half:
    model.half()  # to FP16
if device.type != 'cpu':
    model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(
        next(model.parameters())))  # run once
model.eval()


def detect(im0):
    img = im0.copy()
    img = letterbox(img, imgsz, auto=imgsz != 1280)[0]
    img = img[:, :, ::-1].transpose(2, 0, 1)
    img = np.ascontiguousarray(img)

    with torch.no_grad():
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        t0 = time.perf_counter()

        pred = model(img, augment=args["AUGMENTED_INFERENCE"])[0]
        pred = non_max_suppression(pred,
                                   args["CONF_THRESHOLD"],
                                   args["IOU_THRESHOLD"],
                                   args["CLASSES_TO_DETECT"],
                                   args["AGNOSTIC_NMS"])

        img.detach().cpu()

        inference_time = time.perf_counter()-t0
        
        detected_objects = []
        for det in pred:
            if len(det) > 0:
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
                for *xyxy, confidence, class_index in reversed(det):
                    if int(class_index) in args["CLASSES_TO_DETECT"]:
                        x1 = xyxy[0]
                        y1 = xyxy[1]
                        x2 = xyxy[2]
                        y2 = xyxy[3]
                        width = x2 - x1
                        height = y2 - y1

                        obj = {"x1": int(x1), "y1": int(y1), 
                               "x2": int(x2), "y2": int(y2),
                               "area": float(width * height),
                               "ratio": float(height / width),
                               "x_center": float((x1 + x2) / 2.),
                               "y_center": float((y1 + y2) / 2.),
                               "score": round(float(confidence),2),
                               "label": args["CLASS_NAMES"][int(class_index)],
                               "class_id": int(class_index)}
                        
                        detected_objects.append(obj)

        logger.info("{} Objects - Time: {}".format(len(detected_objects), inference_time))
    
    # img.detach().cpu()    
    # del img
    # torch.cuda.empty_cache()
    # gc.collect()

    return detected_objects


def load_image(base64_image):
    image_base64 = base64_image.split(',', 1)[-1]
    image = Image.open(BytesIO(base64.b64decode(image_base64)))
    im0 = np.array(image)
    height = im0.shape[0]
    width = im0.shape[1]
    return im0, height, width


def on_message(c, userdata, msg):

    msg_time_0 = time.perf_counter()

    message = str(msg.payload.decode("utf-8", "ignore"))
    try:
        data_received = json.loads(message)
    except json.JSONDecodeError as e:
        logger.error("Error decoding JSON:", e)
        return

    img_array, orig_height, orig_width = load_image(data_received["image"])

    detected_objects = detect(img_array)

    msg_time_1 = time.perf_counter()

    payload = {
        "timestamp": data_received["timestamp"],
        "image": data_received["image"],
        "type": "objects",
        "data": detected_objects,
        "metadata": {
            "applicatons": {
                "name": APP_NAME, 
                "version": "v1.0"},
            "peripheral": {
                "id": data_received["peripheral_id"],
                "name": data_received["peripheral_name"],
                "type": data_received["peripheral_type"]},
            "processing": {
                'image_height': orig_height, 
                'image_width': orig_width,
                'runtime': msg_time_1 - msg_time_0}
        }
    }

    msg = json.dumps(payload, cls=NumpyEncoder)
    client.publish(userdata['MQTT_OUT_0'], msg)


if args['MQTT_VERSION'] == '5':
    client = mqtt.Client(client_id=APP_NAME,
                         transport=args['MQTT_TRANSPORT'],
                         protocol=mqtt.MQTTv5,
                         userdata=args)
    client.reconnect_delay_set(min_delay=1, max_delay=120)
    client.on_connect = on_connect_v5
    client.on_message = on_message
    client.connect(args['MQTT_SERVICE_HOST'], port=args['MQTT_SERVICE_PORT'],
                   clean_start=mqtt.MQTT_CLEAN_START_FIRST_ONLY, keepalive=60)

if args['MQTT_VERSION'] == '3':
    client = mqtt.Client(client_id=APP_NAME, transport=args['MQTT_TRANSPORT'],
                         protocol=mqtt.MQTTv311, userdata=args, clean_session=True)
    client.reconnect_delay_set(min_delay=1, max_delay=120)
    client.on_connect = on_connect_v3
    client.on_message = on_message
    client.connect(args['MQTT_SERVICE_HOST'],
                   port=args['MQTT_SERVICE_PORT'], keepalive=60)

client.enable_logger(logger=logger)

# This runs the network code in a background thread and also handles reconnecting for you.
client.loop_forever()
