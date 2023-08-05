#!/usr/bin/env python

import os
import sys
import cv2
import json
import time
import base64
import logging
import torch
import torch.backends.cudnn as cudnn
from io import BytesIO
import numpy as np
from PIL import Image
import paho.mqtt.client as mqtt

import warnings
warnings.filterwarnings('ignore')

# these imports draw from yolov7, which is cloned when the dockerfiles are built
from models.experimental import attempt_load
from utils.general import non_max_suppression, check_img_size, scale_coords
from utils.torch_utils import select_device, time_synchronized
from utils.datasets import letterbox

from tracker.byte_tracker import BYTETracker

APP_NAME = os.getenv('APP_NAME', 'yolov7-bytetrack')

args = {
    'NAME': APP_NAME,

    'MQTT_IN_0': os.getenv("MQTT_IN_0", f"{APP_NAME}/images"),
    'MQTT_OUT_0': os.getenv("MQTT_OUT_0", f"{APP_NAME}/events"),
    'MQTT_VERSION': os.getenv("MQTT_VERSION", '3'),
    'MQTT_TRANSPORT': os.getenv("MQTT_TRANSPORT", 'tcp'),
    'MQTT_SERVICE_HOST': os.getenv('MQTT_SERVICE_HOST', '127.0.0.1'),
    'MQTT_SERVICE_PORT': int(os.getenv('MQTT_SERVICE_PORT', '1883')),

    'DEVICE': os.getenv("DEVICE", '0'),

    # 'MODEL_NAME': os.getenv("MODEL_NAME", "yolov7-coco-bytetrack"),
    # 'MODEL_VERSION': os.getenv("MODEL_VERSION", "0.1"),
    # 'MODEL_ID': os.getenv("MODEL_ID", "abc123"),

    'WEIGHTS': os.getenv("WEIGHTS", "model.pt"),
    'AGNOSTIC_NMS': os.getenv("AGNOSTIC_NMS", ""),
    'IMG_SIZE': int(os.getenv("CONF_THRESHOLD", 640)),
    'CLASS_NAMES': os.getenv("CLASS_NAMES", "obj.names"),
    'IOU_THRESHOLD': float(os.getenv("IOU_THRESHOLD", 0.45)),
    # 'AUGMENTED_INFERENCE': os.getenv("AUGMENTED_INFERENCE", ""),
    'CONF_THRESHOLD': float(os.getenv("CONF_THRESHOLD", 0.25)),
    'CLASSES_TO_DETECT': str(os.getenv("CLASSES_TO_DETECT", "person,bicycle,car,motorbike,truck")),

    "TRACKER_THRESHOLD": float(os.getenv("TRACKER_THRESHOLD", 0.5)),
    "TRACKER_MATCH_THRESHOLD": float(os.getenv("TRACKER_MATCH_THRESOLD", 0.8)),
    "TRACKER_BUFFER": int(os.getenv("TRACKER_BUFFER", 30)),
    "TRACKER_FRAME_RATE": int(os.getenv("TRACKER_FRAME_RATE", 10)),
}

logger = logging.getLogger(args['NAME'])
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


# if args["AUGMENTED_INFERENCE"] == "":
#     args["AUGMENTED_INFERENCE"] = False
# else:
#     args["AUGMENTED_INFERENCE"] = True

if args["AGNOSTIC_NMS"] == "":
    args["AGNOSTIC_NMS"] = False
else:
    args["AGNOSTIC_NMS"] = True

if args["CLASS_NAMES"] != "":
    class_names = []
    with open(args["CLASS_NAMES"], "r", encoding='utf-8') as names_file:
        for line in names_file:
            if line != "" and line != "\n":
                class_names.append(line.strip())
    args["CLASS_NAMES"] = class_names
else:
    logger.info("You must specify 'CLASS_NAMES'")
    logger.info("App Exit!")
    sys.exit(1)

if args["CLASSES_TO_DETECT"] == "":
    args["CLASSES_TO_DETECT"] = None
else:
    cls_to_detect = args["CLASSES_TO_DETECT"]
    if len(cls_to_detect) == 1:
        cls_to_detect = args["CLASS_NAMES"].index(cls_to_detect)
    else:
        cls_to_detect = cls_to_detect.split(",")
        cls_ids = []
        for index, cls_name in enumerate(cls_to_detect):
            cls_id = args["CLASS_NAMES"].index(cls_name)
            cls_ids.append(cls_id)
        args["CLASSES_TO_DETECT"] = cls_ids
        del cls_ids, cls_to_detect


logger.info("Initializing Object Detection Model")
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

tracker = BYTETracker(track_thresh=args["TRACKER_THRESHOLD"],
                      match_thresh=args["TRACKER_MATCH_THRESHOLD"],
                      track_buffer=args["TRACKER_BUFFER"],
                      frame_rate=args["TRACKER_FRAME_RATE"])


def detect_and_track(im0):
    img = im0.copy()
    img = letterbox(img, imgsz, auto=imgsz != 1280)[0]
    #img = cv2.resize(np.array(img), (userdata["IMG_SIZE"], userdata["IMG_SIZE"]))
    img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
    img = np.ascontiguousarray(img)
    # img = np.expand_dims(img, axis=0)
    img = torch.from_numpy(img).to(device)
    img = img.half() if half else img.float()  # uint8 to fp16/32
    img /= 255.0  # 0 - 255 to 0.0 - 1.0
    if img.ndimension() == 3: img = img.unsqueeze(0)
    
    with torch.no_grad():
        t0 = time_synchronized()
        pred = model(img)[0] #, augment=args["AUGMENTED_INFERENCE"])[0]
        pred = non_max_suppression(pred,
                                   args["CONF_THRESHOLD"],
                                   args["IOU_THRESHOLD"],
                                   args["CLASSES_TO_DETECT"],
                                   args["AGNOSTIC_NMS"])
    inference_time = time_synchronized() - t0
    logger.info("YOLOv7 Inference Time : {}".format(inference_time))

    # pred = list of detections, on (n,6) tensor per image [xyxy, conf, cls]
    # loop below from: https://github.com/theos-ai/easy-yolov7/blob/main/algorithm/object_detector.py
    raw_detection = np.empty((0,6), float)
    for det in pred:
        if len(det) > 0:
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
            for *xyxy, conf, cls in reversed(det):
                raw_detection = np.concatenate((raw_detection, [[int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3]), round(float(conf), 2), int(cls)]]))
    
    tracked_objects = tracker.update(raw_detection)

    return tracked_objects


def load_image(base64_image):
    image_base64 = base64_image.split(',', 1)[-1]
    image = Image.open(BytesIO(base64.b64decode(image_base64)))
    return np.array(image)


def on_message(c, userdata, msg):
    #try:
    message = str(msg.payload.decode("utf-8", "ignore"))
    # {“timestamp”: “…”, “image”: <base64_mime>, “camera_id”: “A”, “camera_name”: “…”}
    
    try: 
        data_received = json.loads(message)
    except json.JSONDecodeError as e:
        logger.error("Error decoding JSON:", e)
        return

    msg_time_0 = time_synchronized()

    img = load_image(data_received["image"])

    tracked_objects = detect_and_track(img)

    msg_time_1 = time_synchronized()

    payload = {
        "timestamp": data_received["timestamp"],
        "type": "objects",
        "data": [],
        "metadata": {
            "applicaton": {
                "name": APP_NAME, 
                "version": "v1.0",
                "processing_time": msg_time_1 - msg_time_0,
                "input_parameters": userdata},
            "peripheral": {
                "id": "00001",
                "name": "parking-lot-1", 
                "type": "camera",
                "image_height": img.shape[0], 
                "image_width": img.shape[1],
                "fps": 1},
        },
    }

    for tracked_object in tracked_objects:
        x1 = tracked_object[0]
        y1 = tracked_object[1]
        x2 = tracked_object[2]
        y2 = tracked_object[3]
        track_id = tracked_object[4]
        class_index = int(tracked_object[5])
        score = tracked_object[6]
        payload["data"].append({
            'trk_id': track_id,
            'x1': x1,
            'y1': y1,
            'x2': x2,
            'y2': y2,
            'x_center': int((x1 + x2) / 2),
            'y_center': int((y1 + y2) / 2),
            'width': x2 - x1,
            'height': y2 - y1,
            'ratio': (y2 - y1) / (x2 - x1),
            'score': score,
            'area': x2 * y2,
            'label': args["CLASS_NAMES"][class_index],
            # 'track_time': track_time
        })

    msg = json.dumps(payload, cls=NumpyEncoder)
    client.publish(userdata['MQTT_OUT_0'], msg)
    payload["image"] = "%s... - truncated for logs" % payload["image"][0:32]
    logger.info(payload)

    # except Exception as e:
    #     logger.error(e)
    #     sys.exit(1)


if args['MQTT_VERSION'] == '5':
    client = mqtt.Client(client_id=args['NAME'],
                         transport=args['MQTT_TRANSPORT'],
                         protocol=mqtt.MQTTv5,
                         userdata=args)
    client.reconnect_delay_set(min_delay=1, max_delay=120)
    client.on_connect = on_connect_v5
    client.on_message = on_message
    client.connect(args['MQTT_SERVICE_HOST'], port=args['MQTT_SERVICE_PORT'],
                   clean_start=mqtt.MQTT_CLEAN_START_FIRST_ONLY, keepalive=60)

if args['MQTT_VERSION'] == '3':
    client = mqtt.Client(client_id=args['NAME'], transport=args['MQTT_TRANSPORT'],
                         protocol=mqtt.MQTTv311, userdata=args, clean_session=True)
    client.reconnect_delay_set(min_delay=1, max_delay=120)
    client.on_connect = on_connect_v3
    client.on_message = on_message
    client.connect(args['MQTT_SERVICE_HOST'],
                   port=args['MQTT_SERVICE_PORT'], keepalive=60)

client.enable_logger(logger=logger)
# This runs the network code in a background thread and also handles reconnecting for you.
client.loop_forever()
