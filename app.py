import time
import os
import sys
import logging
import gc

import json
import base64
from io import BytesIO
from PIL import Image

import paho.mqtt.client as mqtt

import numpy as np
import torch
import torch.backends.cudnn as cudnn

from models.experimental import attempt_load
from utils.datasets import letterbox
from utils.general import check_img_size, non_max_suppression, scale_coords, xyxy2xywh, set_logging
from utils.torch_utils import select_device, time_synchronized
from utils.plots import plot_one_box

#  MQTT_IN_0="camera/images" MQTT_SERVICE_HOST=192.168.68.104 MQTT_SERVICE_PORT=31883 WEIGHTS=weights/best_weights.pt IMG_SIZE=640 CLASS_NAMES=ppe-bbox-clean-20220821000000146/dataset/object.names python3 app.py


# This app run Yolov7 deep learning networks.
#
# envs/args:
# APP_NAME:             The app name, will be reflected in logs and client ids (default: yolov7)
# MQTT_SERVICE_HOST:    MQTT broker host ip or hostname (default: mqtt.kube-system)
# MQTT_SERVICE_PORT:    MQTT broker port (default: 1883)
# WEIGHTS:
# CLASS_NAMES:
# CLASSES:
# IMG_SIZE:             (default: 416)
# CONF_THRESHOLD:       (default: 0.25)
# IOU_THRESHOLD:        (default: 0.45)
# DEVICE:               Select cuda device or cpu i.e. 0 or 0,1,2,3 or cpu  (default: cpu)
# AUGMENTED_INFERENCE:
# AGNOSTIC_NMS:
# MODEL_NAME:           Model name for metadata (default: yolov7)
# MQTT_VERSION:         MQTT protocol version 3 or 5 (default: 3)
# MQTT_TRANSPORT:       MQTT protocol transport for version 5, tcp or websockets (default: tcp)

APP_NAME = os.getenv('APP_NAME', 'yolov7')

args = {
    'NAME': APP_NAME,
    'MQTT_SERVICE_HOST': os.getenv('MQTT_SERVICE_HOST', 'mqtt.kube-system'),
    'MQTT_SERVICE_PORT': int(os.getenv('MQTT_SERVICE_PORT', '1883')),
    'MQTT_IN_0': os.getenv("MQTT_IN_0", "camera/images"),
    'MQTT_OUT_0': os.getenv("MQTT_OUT_0", f"{APP_NAME}/events"),
    'WEIGHTS': os.getenv("WEIGHTS", ""),
    'CLASS_NAMES': os.getenv("CLASS_NAMES", ""),
    'CLASSES': os.getenv("CLASSES", ""),
    'IMG_SIZE': int(os.getenv("IMG_SIZE", 416)),
    'CONF_THRESHOLD': float(os.getenv("CONF_THRESHOLD", 0.25)),
    'IOU_THRESHOLD': float(os.getenv("IOU_THRESHOLD", 0.45)),
    'DEVICE': os.getenv("DEVICE", 'cpu'),  # cuda device, i.e. 0 or 0,1,2,3 or cpu
    'AUGMENTED_INFERENCE': os.getenv("AUGMENTED_INFERENCE", ""),
    'AGNOSTIC_NMS': os.getenv("AGNOSTIC_NMS", ""),
    'MODEL_NAME': os.getenv("MODEL_NAME", 'yolov7'),  # define from model config - better to use a registry
    'MQTT_VERSION': os.getenv("MQTT_VERSION", '3'),  # or 5
    'MQTT_TRANSPORT': os.getenv("MQTT_TRANSPORT", 'tcp'),  # or websockets
}

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
    with open(args["CLASS_NAMES"],"r",encoding='utf-8') as names_file:
        for line in names_file:
            if line != "" and line != "\n":
                class_names.append(line.strip())
    args["CLASS_NAMES"] = class_names
else:
    print("You must specify 'CLASS_NAMES'")
    sys.exit(1)

if args["CLASSES"] == "":
    args["CLASSES"] = None

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
    """ Special json encoder for numpy types """

    def default(self, obj):
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


# Initialize
set_logging()
device = select_device(args["DEVICE"])
half = device.type != 'cpu'  # half precision only supported on CUDA

# Load model
model = attempt_load(args["WEIGHTS"], map_location=device)  # load FP32 model
stride = int(model.stride.max())  # model stride
imgsz = args["IMG_SIZE"]
if isinstance(imgsz, (list, tuple)):
    assert len(imgsz) == 2
    "height and width of image has to be specified"
    imgsz[0] = check_img_size(imgsz[0], s=stride)
    imgsz[1] = check_img_size(imgsz[1], s=stride)
else:
    imgsz = check_img_size(imgsz, s=stride)  # check img_size
# names = model.module.names if hasattr(model, 'module') else model.names  # get class names
if half:
    model.half()  # to FP16

# Run inference
if device.type != 'cpu':
    model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(
        next(model.parameters())))  # run once

model.eval()

args["MODEL"] = model
args["STRIDE"] = stride


def detect(userdata, im0, image_mime):
    # Padded resize
    img = letterbox(im0, userdata["IMG_SIZE"], stride=userdata["STRIDE"])[0]

    # Convert
    img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
    img = np.ascontiguousarray(img)

    img = np.expand_dims(img, axis=0)

    t0 = time.time()
    img_tensor = torch.from_numpy(img).to(device)
    img_tensor = img_tensor.half() if half else img_tensor.float()  # uint8 to fp16/32
    img_tensor /= 255.0  # 0 - 255 to 0.0 - 1.0

    # Inference
    with torch.no_grad():

        t1 = time_synchronized()
        pred = userdata['MODEL'](img_tensor, augment=userdata["AUGMENTED_INFERENCE"])[0]
        pred = non_max_suppression(pred,
                                userdata["CONF_THRESHOLD"],
                                userdata["IOU_THRESHOLD"],
                                classes=userdata["CLASSES"],
                                agnostic=userdata["AGNOSTIC_NMS"])
        t2 = time_synchronized()

        detections = []
        for i, det in enumerate(pred):

            # in case there are no predictions - eventually move "del" statement up in the code to avoid this
            gn = None
            conf = None
            xyxy = None
            xywh = None
            label_index = None
            label = None
            label_index = None
            confidence = None
            detected_label = None

            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img_tensor.shape[2:], det[:, :4], im0.shape).round()

                gn = torch.tensor(img_tensor.shape)[[1, 0, 1, 0]]  # normalization gain whwh

                object_id = 0
                for *xyxy, confidence, detected_label in reversed(det):
                    xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) ).view(-1).tolist()  # normalized xywh
                    conf = confidence.item()
                    
                    label_index = int(detected_label.item())                
                    label=None
                    if label_index >= 0 and label_index < len(userdata["CLASS_NAMES"]):
                        label = userdata["CLASS_NAMES"][label_index]

                    if label:
                        detections.append({'objId': object_id,
                                            'id': object_id,
                                            'nx': im0.shape[1],
                                            'ny': im0.shape[0],
                                            'bbox': [(xywh[0]-xywh[2]/2.)/im0.shape[1],
                                                     (xywh[1]-xywh[3]/2.)/im0.shape[0],
                                                     (xywh[2])/im0.shape[1],
                                                     (xywh[3])/im0.shape[0]],
                                            'className': label,
                                            'label': label,
                                            'xmin': int(xywh[0]-xywh[2]/2.),
                                            'ymin': int(xywh[1]-xywh[3]/2.),
                                            'width': xywh[2],
                                            'height': xywh[3],
                                            'xmax': int((xywh[0]-xywh[2]/2.)+xywh[2]),
                                            'ymax': int((xywh[1]-xywh[3]/2.)+xywh[3]),
                                            'area': xywh[2]*xywh[3],
                                            'score': conf})
                        object_id += 1
    
        payload = {
            "model": userdata["MODEL_NAME"],
            "image": image_mime,
            "inference_time": t2 - t1,
            "objects": detections
        }

        msg = json.dumps(payload, cls=NumpyEncoder)
        client.publish(userdata['MQTT_OUT_0'], msg)
        payload["image"] = "%s... - truncated for logs" % payload["image"][0:32]
        logger.info(payload)

    del payload, detections, img_tensor, gn, conf, label, xywh, xyxy, pred, img, t0, t1, t2, confidence, detected_label, label_index, det
    gc.collect()
    torch.cuda.empty_cache()


def on_message(c, userdata, msg):
    try:
        image_mime = str(msg.payload.decode("utf-8", "ignore"))
        _, image_base64 = image_mime.split(',', 1)
        image = Image.open(BytesIO(base64.b64decode(image_base64)))
        detect(userdata, im0=np.array(image), image_mime=image_mime)
    except Exception as e:
        logger.error('Error:', e)
        exit(1)


if args['MQTT_VERSION'] == '5':
    client = mqtt.Client(client_id=args['NAME'],
                         transport=args['MQTT_TRANSPORT'],
                         protocol=mqtt.MQTTv5,
                         userdata=args)
    client.reconnect_delay_set(min_delay=1, max_delay=120)
    client.on_connect = on_connect_v5
    client.on_message = on_message
    client.connect(args['MQTT_SERVICE_HOST'],
                   port=args['MQTT_SERVICE_PORT'],
                   clean_start=mqtt.MQTT_CLEAN_START_FIRST_ONLY,
                   keepalive=60)

if args['MQTT_VERSION'] == '3':
    client = mqtt.Client(client_id=args['NAME'],
                         transport=args['MQTT_TRANSPORT'],
                         protocol=mqtt.MQTTv311,
                         userdata=args,
                         clean_session=True)
    client.reconnect_delay_set(min_delay=1, max_delay=120)
    client.on_connect = on_connect_v3
    client.on_message = on_message
    client.connect(args['MQTT_SERVICE_HOST'], port=args['MQTT_SERVICE_PORT'], keepalive=60)

client.enable_logger(logger=logger)
# This runs the network code in a background thread and also handles reconnecting for you.
client.loop_forever()
