#!/usr/bin/env python

# Objective: Run object detection on images from multiple cameras simultaneously
# Hurdle: Error indicating that data is full precision but model is half when running on GPU

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

# these imports draw from yolov7, which is cloned when the dockerfiles are built
from models.experimental import attempt_load
from utils.general import non_max_suppression, check_img_size
from utils.torch_utils import select_device

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

    'MODEL_NAME': os.getenv("MODEL_NAME", "yolov7-coco-bytetrack"),
    'MODEL_VERSION': os.getenv("MODEL_VERSION", "0.1"),
    'MODEL_ID': os.getenv("MODEL_ID", "abc123"),

    'WEIGHTS': os.getenv("WEIGHTS", "model.pt"),
    'AGNOSTIC_NMS': os.getenv("AGNOSTIC_NMS", ""),
    'IMG_SIZE': int(os.getenv("CONF_THRESHOLD", 640)),
    'CLASS_NAMES': os.getenv("CLASS_NAMES", "obj.names"),
    'IOU_THRESHOLD': float(os.getenv("IOU_THRESHOLD", 0.45)),
    'AUGMENTED_INFERENCE': os.getenv("AUGMENTED_INFERENCE", ""),
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


# Resetting the User Arguments
if args["AUGMENTED_INFERENCE"] == "":
    args["AUGMENTED_INFERENCE"] = False
else:
    args["AUGMENTED_INFERENCE"] = True

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


class ObjectTrackers:

    def __init__(self, args):
        self._trackers = {}
        self.threshold = args["TRACKER_THRESHOLD"]
        self.match_threshold = args["TRACKER_MATCH_THRESHOLD"]
        self.buffer = args["TRACKER_BUFFER"]
        self.frame_rate = args["TRACKER_FRAME_RATE"]

    def _new_tracker(self):
        # should we consider allowing the input payload to dynamically
        # define the frame rate, etc. instead of assuming it here?
        return BYTETracker(track_thresh=self.threshold,
                           match_thresh=self.match_threshold,
                           track_buffer=self.buffer,
                           frame_rate=self.frame_rate)

    def get(self, camera_id):
        if camera_id not in self._trackers:
            self._trackers[camera_id] = self._new_tracker()
        return self._trackers[camera_id]


logger.info("Tracking Model Initialization")
trackers = ObjectTrackers(args)


def detect(image_stack):
    t0 = time.time()
    with torch.no_grad():
        model_input = torch.from_numpy(image_stack).to(device)
        model_input = model_input.half() if half else model_input.float()  # uint8 to fp16/32
        model_input /= 255.0  # 0 - 255 to 0.0 - 1.0
        model_output = model(model_input, augment=args["AUGMENTED_INFERENCE"])[0]
        detections = non_max_suppression(
            model_output,
            args["CONF_THRESHOLD"],
            args["IOU_THRESHOLD"],
            args["CLASSES_TO_DETECT"],
            args["AGNOSTIC_NMS"])
    inference_time = time.time() - t0
    logger.info("YOLOv7 Inference Time : {}".format(inference_time))
    return detections, inference_time


def load_image(base64_image, userdata):
    # preprocess of base64 string
    image_base64 = base64_image.split(',', 1)[-1]
    # Decode the base64 string to byte64 array and read image
    image = Image.open(BytesIO(base64.b64decode(image_base64)))
    # Convert the BGR image to RGB
    image = cv2.cvtColor(np.array(image), cv2.COLOR_BGR2RGB)
    # Resize an image to model size
    image = cv2.resize(image, (userdata["IMG_SIZE"], userdata["IMG_SIZE"]))
    return np.ascontiguousarray(image)


def on_message(c, userdata, msg):
    try:
        message = str(msg.payload.decode("utf-8", "ignore"))
        # {"images": [{“timestamp”: “…”, “image”: <base64_mime>, “camera_id”: “A”, “camera_name”: “…”}, ...]}
        try:
            data_received = json.loads(message)
        except json.JSONDecodeError as e:
            logger.error("Error decoding JSON:", e)
            sys.exit(1)

        camera_images = data_received["images"]

        images = []  # camera_image["image"] for camera_image in camera_images]
        for camera_image in camera_images:
            img = load_image(camera_image["image"], userdata)
            images.append(img)

        # Represent list of images as a stack (tensor)
        image_stack = np.transpose(
            np.flip(np.stack(images), 3), (0, 3, 1, 2)).astype(np.float32) / 255.0
        
        image_stack = np.ascontiguousarray(image_stack)

        detections, inference_time = detect(image_stack)

        payload = {
            # from registry eventually
            "model": {"name": args["MODEL_NAME"],
                      "version": args["MODEL_VERSION"],
                      "id": args["MODEL_ID"]},
            "inference_time": inference_time,
            "results": []
        }

        for i, (camera_image, detection) in enumerate(zip(camera_images, detections)):

            result = {}

            result["timestamp"] = data_received["timestamp"]

            result["data"] = camera_image
            result["data"]["width"] = images[i].shape[1]
            result["data"]["height"] = images[i].shape[0]

            camera_id = camera_image["camera_id"]
            tracker = trackers.get(camera_id)
            track_time_0 = time.time()
            tracked_objects = tracker.update_multi(
                torch.tensor(detection), images[i])
            track_time = time.time() - track_time_0

            result["objects"] = []
            for tracked_object in tracked_objects:
                x1 = tracked_object[0]
                y1 = tracked_object[1]
                x2 = tracked_object[2]
                y2 = tracked_object[3]
                track_id = tracked_object[4]
                class_index = int(tracked_object[5])
                score = tracked_object[6]

                result["objects"].append({
                    'trk_id': track_id,
                    'x1': x1,
                    'y1': y1,
                    'x2': x2,
                    'y2': y2,
                    'xc': int((x1 + x2) / 2),
                    'yc': int((y1 + y2) / 2),
                    'width': x2 - x1,
                    'height': y2 - y1,
                    'ratio': (y2 - y1) / (x2 - x1),
                    'score': score,
                    'area': x2 * y2,
                    'label': args["CLASS_NAMES"][class_index],
                    'track_time': track_time
                })

            payload["results"].append(result)

        msg = json.dumps(payload, cls=NumpyEncoder)
        client.publish(userdata['MQTT_OUT_0'], msg)

        # for result in payload["results"]:
        #     for obj in result["data"]:
        #         if "image" in obj:
        #             obj["image"]="{} - truncated for logs".format(
        #                 obj["image"][:32])
        # logger.debug(payload)

    except Exception as e:
        logger.error('Error:', e)
        sys.exit(1)


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
