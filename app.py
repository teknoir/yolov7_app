#!/usr/bin/env python

import os
import sys
import json
import time
import datetime
import base64
import gc
import logging
import torch
import torch.backends.cudnn as cudnn
import numpy as np
import paho.mqtt.client as mqtt
from math import dist
from io import BytesIO
from PIL import Image

from tracker.byte_tracker import BYTETracker

# these imports draw from yolov7, which is cloned when the dockerfiles are built
from utils.datasets import letterbox
from utils.torch_utils import select_device, time_synchronized
from utils.general import non_max_suppression, check_img_size, scale_coords
from models.experimental import attempt_load

import warnings
warnings.filterwarnings('ignore')


APP_NAME = os.getenv('APP_NAME', 'yolov7-bytetrack')

args = {
    'NAME': APP_NAME,

    'MQTT_IN_0': os.getenv("MQTT_IN_0", f"{APP_NAME}/images"),
    'MQTT_OUT_0': os.getenv("MQTT_OUT_0", f"{APP_NAME}/detections"),
    'MQTT_OUT_1': os.getenv("MQTT_OUT_0", f"{APP_NAME}/movements"),
    'MQTT_VERSION': os.getenv("MQTT_VERSION", '3'),
    'MQTT_TRANSPORT': os.getenv("MQTT_TRANSPORT", 'tcp'),
    'MQTT_SERVICE_HOST': os.getenv('MQTT_SERVICE_HOST', '127.0.0.1'),
    'MQTT_SERVICE_PORT': int(os.getenv('MQTT_SERVICE_PORT', '1883')),

    'DEVICE': os.getenv("DEVICE", '0'),

    # 'SHOW_DEBUG': bool(os.getenv("SHOW_DEBUG", 'False')),

    # all of these model params are tied to the application container
    # 'MODEL_NAME': os.getenv("MODEL_NAME", "yolov7-coco-bytetrack"),
    # 'MODEL_VERSION': os.getenv("MODEL_VERSION", "0.1"),
    # 'MODEL_ID': os.getenv("MODEL_ID", "abc123"),

    'WEIGHTS': str(os.getenv("WEIGHTS", "model.pt")),
    'AGNOSTIC_NMS': os.getenv("AGNOSTIC_NMS", ""),
    'IMG_SIZE': int(os.getenv("CONF_THRESHOLD", "640")),
    'CLASS_NAMES': os.getenv("CLASS_NAMES", "obj.names"),
    'IOU_THRESHOLD': float(os.getenv("IOU_THRESHOLD", "0.45")),
    'CONF_THRESHOLD': float(os.getenv("CONF_THRESHOLD", "0.25")),
    # 'AUGMENTED_INFERENCE': os.getenv("AUGMENTED_INFERENCE", ""),
    'CLASSES_TO_DETECT': str(os.getenv("CLASSES_TO_DETECT", "person,bicycle,car,motorbike,truck")),

    "TRACKER_THRESHOLD": float(os.getenv("TRACKER_THRESHOLD", "0.5")),
    "TRACKER_MATCH_THRESHOLD": float(os.getenv("TRACKER_MATCH_THRESOLD", "0.8")),
    "TRACKER_BUFFER": int(os.getenv("TRACKER_BUFFER", "30")),
    "TRACKER_FRAME_RATE": int(os.getenv("TRACKER_FRAME_RATE", "10")),

    "EXIT_AFTER_SECONDS": float(os.getenv("EXIT_AFTER_SECONDS", 5.0))
}

logger = logging.getLogger(args['NAME'])
ch = logging.StreamHandler(sys.stdout)
# if args["SHOW_DEBUG"] == True:
#     ch.setLevel(logging.DEBUG)
# else:
ch.setLevel(logging.INFO)
formatter = logging.Formatter(
    '[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)

# if args["SHOW_DEBUG"] == True:
#     logger.setLevel(logging.DEBUG)
# else:
logger.setLevel(logging.INFO)

logger.info("TΞꓘN01R")
logger.info("TΞꓘN01R")
logger.info("TΞꓘN01R")

logger.info(json.dumps(args))

# def log_gpu_memory_usage(human_readable_code_location):
#     debug_msg = "{}: Mem Allocated {} - Mem Reserved {} - GC Objects {}".format(
#         human_readable_code_location,
#         torch.cuda.memory_allocated(),
#         torch.cuda.memory_reserved(),
#         gc.get_count())
#     logger.debug(debug_msg)


# log_gpu_memory_usage("init")


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

# log_gpu_memory_usage("model")

tracker = BYTETracker(track_thresh=args["TRACKER_THRESHOLD"],
                      match_thresh=args["TRACKER_MATCH_THRESHOLD"],
                      track_buffer=args["TRACKER_BUFFER"],
                      frame_rate=args["TRACKER_FRAME_RATE"])


def detect_and_track(im0):

    # log_gpu_memory_usage("detect_init")

    img = im0.copy()
    img = letterbox(img, imgsz, auto=imgsz != 1280)[0]
    #img = cv2.resize(np.array(img), (userdata["IMG_SIZE"], userdata["IMG_SIZE"]))
    img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
    img = np.ascontiguousarray(img)
    # img = np.expand_dims(img, axis=0)

    img = torch.from_numpy(img).to(device)
    img = img.half() if half else img.float()  # uint8 to fp16/32
    img /= 255.0  # 0 - 255 to 0.0 - 1.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)

    # log_gpu_memory_usage("img_loaded")

    t0 = time_synchronized()
    with torch.no_grad():
        pred = model(img)[0]  # , augment=args["AUGMENTED_INFERENCE"])[0]
        pred = non_max_suppression(pred,
                                   args["CONF_THRESHOLD"],
                                   args["IOU_THRESHOLD"],
                                   args["CLASSES_TO_DETECT"],
                                   args["AGNOSTIC_NMS"])

    # pred = list of detections, on (n,6) tensor per image [xyxy, conf, cls]
    # loop below from: https://github.com/theos-ai/easy-yolov7/blob/main/algorithm/object_detector.py
    raw_detection = np.empty((0, 6), float)
    for det in pred:
        if len(det) > 0:
            det[:, :4] = scale_coords(
                img.shape[2:], det[:, :4], im0.shape).round()
            for *xyxy, conf, cls in reversed(det):
                raw_detection = np.concatenate((raw_detection, [[int(xyxy[0]), int(
                    xyxy[1]), int(xyxy[2]), int(xyxy[3]), round(float(conf), 2), int(cls)]]))

    tracked_objects = tracker.update(raw_detection)

    logger.info(
        "YOLOv7 + ByteTrack Inference Time : {}".format(time_synchronized()-t0))

    # log_gpu_memory_usage("tracked")

    img.detach().cpu()

    # log_gpu_memory_usage("img_detach")

    del img  # delete from allocated memory

    torch.cuda.empty_cache()  # delete from reserved memory
    gc.collect()

    # log_gpu_memory_usage("del_img")

    return tracked_objects


def load_image(base64_image):
    image_base64 = base64_image.split(',', 1)[-1]
    image = Image.open(BytesIO(base64.b64decode(image_base64)))
    return np.array(image)


class TrackedObjectsBuffer:

    def __init__(self):
        self.objects = {}

    def update(self, obj):
        if obj["id"] not in self.objects:
            self._enter_object(obj)
        else:
            self._append_object(obj)

    def _append_object(self, obj):
        self.objects[obj["id"]].append(obj)

    def _enter_object(self, obj):
        logger.info(f"ENTER: {obj['label']} - {obj['id']}")
        self.objects[obj["id"]] = [obj]
    
    def _exit_object(self, obj_id):
        movement = {}
        movement["id"] = obj_id
        movement["start_time"] = self.objects[obj_id][0]["timestamp"]
        movement["end_time"] = self.objects[obj_id][-1]["timestamp"]
        # ASSUMPTION: timestamps are in javascript Date.now() format.
        end = datetime.datetime.fromtimestamp(int(movement["end_time"])/1000.0)
        start = datetime.datetime.fromtimestamp(
            int(movement["start_time"])/1000.0)
        movement["duration"] = (end - start).total_seconds()
        movement["start_time_iso"] = start.isoformat()
        movement["end_time_iso"] = end.isoformat()

        labels = [obj["label"] for obj in self.objects[obj_id]]
        movement["label"] = max(set(labels), key=labels.count)
        movement["labels"] = labels
        
        movement["length"] = len(self.objects[obj_id])

        movement["width_average"] = np.mean(
            [obj["width"] for obj in self.objects[obj_id]])
        movement["height_average"] = np.mean(
            [obj["height"] for obj in self.objects[obj_id]])
        movement["ratio_average"] = np.mean(
            [obj["ratio"] for obj in self.objects[obj_id]])
        movement["area_average"] = np.mean(
            [obj['area'] for obj in self.objects[obj_id]])
        movement["score_average"] = np.mean(
            [obj['score'] for obj in self.objects[obj_id]])

        movement["trajectory"] = [(obj["x_center"], obj["y_center"])
                                  for obj in self.objects[obj_id]]
        movement["history"] = self.objects[obj_id]

        movement["metadata"] = {
            "parameters": args,
            "applicaton": {
                "name": APP_NAME,
                "version": "v1.0"}
        }

        msg = json.dumps(movement, cls=NumpyEncoder)
        client.publish(args["MQTT_OUT_1"], msg)

        logger.info(f"EXIT: {movement['label']} - {obj_id}")

        del self.objects[obj_id]

    def monitor(self, current_time):
        now = datetime.datetime.fromtimestamp(int(current_time)/1000.0)
        for obj_id in list(self.objects):
            last_updated = self.objects[obj_id][-1]["timestamp"]
            then = datetime.datetime.fromtimestamp(int(last_updated)/1000.0)
            if (now - then) > datetime.timedelta(seconds=args["EXIT_AFTER_SECONDS"]):
                self._exit_object(obj_id)


tracked_objects_buffer = TrackedObjectsBuffer()


def on_message(c, userdata, msg):
    message = str(msg.payload.decode("utf-8", "ignore"))
    # {"timestamp": "<js_epoch_time>", "image": "<base64_mime>", "camera_id": "...", "camera_name": "..."}

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
        "image": data_received["image"],
        "type": "objects",
        "data": [],
        "metadata": {
            "applicaton": {
                "name": APP_NAME,
                "version": "v1.0",
                "processing_time": msg_time_1 - msg_time_0},
            "peripheral": {
                "id": "00001",
                "name": "parking-lot-1",
                "type": "camera",
                "image_height": img.shape[0],
                "image_width": img.shape[1],
                "camera_fps": 1},
        },
    }

    for tracked_object in tracked_objects:
        x1 = tracked_object[0]
        y1 = tracked_object[1]
        x2 = tracked_object[2]
        y2 = tracked_object[3]
        id = tracked_object[4]
        class_index = int(tracked_object[5])
        score = tracked_object[6]

        tracked_obj = {
            'timestamp': data_received["timestamp"],
            'id': id,
            'x1': x1,
            'y1': y1,
            'x2': x2,
            'y2': y2,
            'x_center': (x1 + x2) / 2,
            'y_center': (y1 + y2) / 2,
            'width': x2 - x1,
            'height': y2 - y1,
            'ratio': (y2 - y1) / (x2 - x1),
            'score': score,
            'area': x2 * y2,
            'label': args["CLASS_NAMES"][class_index]
        }

        payload["data"].append(tracked_obj)

    # Why not vectorized? The overhead of reformatting this would limit performance gains.
    for i, p in enumerate(payload['data']):
        payload['data'][i]['distances'] = {}
        p_point = [p["x_center"], p["y_center"]]
        for q in payload['data']:
            if q["id"] != p["id"]:
                q_point = [q["x_center"], q["y_center"]]
                payload['data'][i]['distances'][q["id"]] = dist(
                    p_point, q_point)

        tracked_objects_buffer.update(payload['data'][i])

    msg = json.dumps(payload, cls=NumpyEncoder)
    client.publish(userdata['MQTT_OUT_0'], msg)

    # logger.debug("{}: {} Objects".format(
    #     time.perf_counter(), len(payload["data"])))
    # logger.info(payload)

    # consider multi-threading this on or placing on a timeloop
    # if timeloop, note that this is comparing data acquisition timestamps
    # this could also be a secondary app, but we'd need to monitor the historian on a loop instead
    tracked_objects_buffer.monitor(payload["timestamp"])


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
