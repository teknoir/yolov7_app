#!/usr/bin/env python

import os
import sys
import json
import time
import math
import base64
import logging
import torch
import torch.backends.cudnn as cudnn
from datetime import timezone, datetime
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
APP_VERSION = os.getenv('APP_VERSION', '0.1.0')

args = {
    "APP_NAME": APP_NAME,
    "APP_VERSION": APP_VERSION,
    'MQTT_IN_0': os.getenv("MQTT_IN_0", f"{APP_NAME}/images"),
    'MQTT_OUT_0': os.getenv("MQTT_OUT_0", f"{APP_NAME}/events"),
    'MQTT_VERSION': os.getenv("MQTT_VERSION", '3'),
    'MQTT_TRANSPORT': os.getenv("MQTT_TRANSPORT", 'tcp'),
    'MQTT_SERVICE_HOST': os.getenv('MQTT_SERVICE_HOST', '127.0.0.1'),
    'MQTT_SERVICE_PORT': int(os.getenv('MQTT_SERVICE_PORT', '1883')),
    'DEVICE': os.getenv("DEVICE", '0'),
    'WEIGHTS': str(os.getenv("WEIGHTS", "model.pt")),
    'AGNOSTIC_NMS': os.getenv("AGNOSTIC_NMS", ""),
    'IMG_SIZE': int(os.getenv("CONF_THRESHOLD", "640")),
    'CLASS_NAMES': os.getenv("CLASS_NAMES", "obj.names"),
    'IOU_THRESHOLD': float(os.getenv("IOU_THRESHOLD", "0.45")),
    'CONF_THRESHOLD': float(os.getenv("CONF_THRESHOLD", "0.25")),
    'AUGMENTED_INFERENCE': os.getenv("AUGMENTED_INFERENCE", ""),
    'CLASSES_TO_DETECT': str(os.getenv("CLASSES_TO_DETECT", "person,bicycle,car,motorbike,truck")),
    "TRACKER_THRESHOLD": float(os.getenv("TRACKER_THRESHOLD", "0.5")),
    "TRACKER_MATCH_THRESHOLD": float(os.getenv("TRACKER_MATCH_THRESOLD", "0.8")),
    "TRACKER_BUFFER": int(os.getenv("TRACKER_BUFFER", "30")),
    "TRACKER_FRAME_RATE": int(os.getenv("TRACKER_FRAME_RATE", "10"))
}

logger = logging.getLogger(args['APP_NAME'])
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


# def base64_encode(ndarray_image):
#     buff = BytesIO()
#     Image.fromarray(ndarray_image).save(buff, format='JPEG')
#     string_encoded = base64.b64encode(buff.getvalue()).decode("utf-8")
#     return f"data:image/jpeg;base64,{string_encoded}"


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


if args["AUGMENTED_INFERENCE"] == "" or args["AUGMENTED_INFERENCE"].lower() == "false":
    args["AUGMENTED_INFERENCE"] = False
else:
    args["AUGMENTED_INFERENCE"] = True

if args["AGNOSTIC_NMS"] == "" or args["AGNOSTIC_NMS"].lower() == "false":
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
    logger.error("You must specify 'CLASS_NAMES'")
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


tracker = BYTETracker(track_thresh=args["TRACKER_THRESHOLD"],
                      match_thresh=args["TRACKER_MATCH_THRESHOLD"],
                      track_buffer=args["TRACKER_BUFFER"],
                      frame_rate=args["TRACKER_FRAME_RATE"])


def detect_and_track(im0):
    img = im0.copy()
    img = letterbox(img, imgsz, auto=imgsz != 1280)[0]
    img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
    img = np.ascontiguousarray(img)

    img = torch.from_numpy(img).to(device)
    img = img.half() if half else img.float()  # uint8 to fp16/32
    img /= 255.0  # 0 - 255 to 0.0 - 1.0
    if img.ndimension() == 3: 
        img = img.unsqueeze(0)

    t0 = time.perf_counter()

    with torch.no_grad():
        pred = model(img)[0] #, augment=args["AUGMENTED_INFERENCE"])[0]
        pred = non_max_suppression(pred,
                                   args["CONF_THRESHOLD"],
                                   args["IOU_THRESHOLD"],
                                   args["CLASSES_TO_DETECT"],
                                   args["AGNOSTIC_NMS"])

    img.detach().cpu()

    # pred = list of detections, on (n,6) tensor per image [xyxy, conf, cls]
    raw_detection = np.empty((0,6), float)
    for det in pred:
        if len(det) > 0:
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
            for *xyxy, conf, cls in reversed(det):
                raw_detection = np.concatenate((raw_detection, [[int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3]), round(float(conf), 2), int(cls)]]))
    
    tracked_objects = tracker.update(raw_detection)

    inference_time = time.perf_counter()-t0

    logger.info("{} Objects - Time: {}".format(
        len(tracked_objects), inference_time))

    return tracked_objects


def calculate_proximities(detections):
    detections_with_proximities = []
    for i, obj1 in enumerate(detections):
        proximity = []
        closest_by_label = {}
        det1 = obj1["detection"]
        p = [det1["x_center"], det1["y_center"]]
        for j, obj2 in enumerate(detections):
            if i != j:
                det2 = obj2["detection"]
                q = [det2["x_center"], det2["y_center"]]
                prox = {"label": det2["label"], 
                        "id": det2["id"],
                        "x_center": det2["x_center"],
                        "y_center": det2["y_center"],
                        "distance": math.dist(p,q)}
                if prox["label"] in closest_by_label:
                    if prox["distance"] < closest_by_label[prox["label"]]["distance"]:
                        closest_by_label[prox["label"]] = prox
                else:
                    closest_by_label[prox["label"]] = prox
                proximity.append(prox)
        obj1["proximity"] = proximity
        obj1["nearest_neighbors"] = closest_by_label
        detections_with_proximities.append(obj1)
    return detections_with_proximities


def load_image(base64_image):
    image_base64 = base64_image.split(',', 1)[-1]
    image = Image.open(BytesIO(base64.b64decode(image_base64)))
    im0 = np.array(image)
    height = im0.shape[0]
    width = im0.shape[1]
    return im0, height, width


def on_message(c, userdata, msg):
    message = str(msg.payload.decode("utf-8", "ignore"))
    # payload: {“timestamp”: “…”, “image”: <base64_mime>, “camera_id”: “A”, “camera_name”: “…”}
    
    try: 
        data_received = json.loads(message)
    except json.JSONDecodeError as e:
        logger.error("Error decoding JSON:", e)
        return
    
    if "image" not in data_received:
        logger.error("No Image. Exiting.")
        return
    
    if "location" not in data_received:
        logger.warning("No Location. Proceeding.")
        data_received["location"] = {"country": "",
                                     "region": "",
                                     "site": "",
                                     "zone": "",
                                     "group": ""}
    
    if "timestamp" not in data_received:
        logger.warning("No timestamp. Using current time.")
        data_received["timestamp"] = int(datetime.now(tz=timezone.utc).timestamp() * 1000)

    msg_time_0 = time.perf_counter()

    try:
        img, orig_height, orig_width = load_image(data_received["image"])
    except Exception as e:
        logger.error(f"Could not load image. Error: {e}")
        return
    
    tracked_objects = detect_and_track(img)

    runtime = time.perf_counter() - msg_time_0

    base_payload = {
            "timestamp": data_received["timestamp"],
            "location": data_received["location"],
            "type": "object"
        }

    if "peripheral" in data_received:
        base_payload["peripheral"] = data_received["peripheral"]

    if "lineage" in data_received:
        base_payload["lineage"] = data_received["lineage"]
    else:
        base_payload["lineage"] = []

    base_payload["lineage"].append({"name": APP_NAME,
                                    "version": APP_VERSION, 
                                    "runtime": runtime})

    detections = []
    for trk in tracked_objects:
        detection_event = base_payload.copy()

        obj = {}
        obj["id"] = str(trk[4])
        obj["x1"] = float(int(trk[0]) / orig_width)
        obj["y1"] = float(int(trk[1]) / orig_height)
        obj["x2"] = float(int(trk[2]) / orig_width)
        obj["y2"] = float(int(trk[3]) / orig_height)
        obj["width"] = float(obj["x2"] - obj["x1"])
        obj["height"] = float(obj["y2"] - obj["y1"])
        obj["area"] = float(obj["height"] * obj["width"])
        obj["ratio"] = float(obj["height"] / obj["width"])
        obj["x_center"] = float((obj["x1"] + obj["x2"])/2.)
        obj["y_center"] = float((obj["y1"] + obj["y2"])/2.)
        obj["score"] = float(trk[6])
        obj["class_id"] = int(trk[5])
        obj["label"] = args["CLASS_NAMES"][obj["class_id"]]

        detection_event["detection"] = obj

        detections.append(detection_event)

    detections = calculate_proximities(detections)

    output = base_payload.copy() # copy everything for frontend, even if no detections
    output["detections"] = detections
    output["image"] = data_received["image"]

    msg = json.dumps(output, cls=NumpyEncoder)
    client.publish(userdata['MQTT_OUT_0'], msg)


if args['MQTT_VERSION'] == '5':
    client = mqtt.Client(client_id=args['APP_NAME'],
                         transport=args['MQTT_TRANSPORT'],
                         protocol=mqtt.MQTTv5,
                         userdata=args)
    client.reconnect_delay_set(min_delay=1, max_delay=120)
    client.on_connect = on_connect_v5
    client.on_message = on_message
    client.connect(args['MQTT_SERVICE_HOST'], port=args['MQTT_SERVICE_PORT'],
                   clean_start=mqtt.MQTT_CLEAN_START_FIRST_ONLY, keepalive=60)

if args['MQTT_VERSION'] == '3':
    client = mqtt.Client(client_id=args['APP_NAME'], transport=args['MQTT_TRANSPORT'],
                         protocol=mqtt.MQTTv311, userdata=args, clean_session=True)
    client.reconnect_delay_set(min_delay=1, max_delay=120)
    client.on_connect = on_connect_v3
    client.on_message = on_message
    client.connect(args['MQTT_SERVICE_HOST'],
                   port=args['MQTT_SERVICE_PORT'], keepalive=60)

client.enable_logger(logger=logger)

# This runs the network code in a background thread and also handles reconnecting for you.
client.loop_forever()

