import time
import os
import sys
import logging

import json
import base64
from io import BytesIO
from PIL import Image

import paho.mqtt.client as mqtt

import numpy as np
import torch
import cv2
from torchvision import transforms
from utils.datasets import letterbox
from utils.general import non_max_suppression_kpt
from utils.plots import output_to_keypoint, plot_skeleton_kpts

#  MQTT_IN_0="camera/images" MQTT_SERVICE_HOST=192.168.68.104 MQTT_SERVICE_PORT=31883 WEIGHTS=weights/best_weights.pt IMG_SIZE=640 CLASS_NAMES=ppe-bbox-clean-20220821000000146/dataset/object.names python3 app.py 


APP_NAME = os.getenv('APP_NAME', 'yolov7-keypoint')

args = {
    'NAME': APP_NAME,
    'MQTT_SERVICE_HOST': os.getenv('MQTT_SERVICE_HOST', 'mqtt.kube-system'),
    'MQTT_SERVICE_PORT': int(os.getenv('MQTT_SERVICE_PORT', '1883')),
    'MQTT_IN_0': os.getenv("MQTT_IN_0", "camera/images"),
    'MQTT_OUT_0': os.getenv("MQTT_OUT_0", f"{APP_NAME}/events"),
    'WEIGHTS': os.getenv("WEIGHTS", "datasets/yolo-pose/yolov7-w6-pose.pt"),
    'TRAINING_DATASET': os.getenv("TRAINING_DATASET", ""),  # define from model config - better to use a registry
    'IMG_SIZE': int(os.getenv("IMG_SIZE", 960)),
    'STRIDE': int(os.getenv("STRIDE", 64)),
    'CONF_THRESHOLD': float(os.getenv("CONF_THRESHOLD", 0.25)),
    'IOU_THRESHOLD': float(os.getenv("IOU_THRESHOLD", 0.65)),
    'DEVICE': os.getenv("DEVICE", 'cpu'),  # cuda device, i.e. 0 or 0,1,2,3 or cpu
    'MODEL_NAME': 'yolov7-keypoint'  # define from model config - better to use a registry
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


def error_str(rc):
    return '{}: {}'.format(rc, mqtt.error_string(rc))


def on_connect(_client, _userdata, _flags, rc):
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


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
weights = torch.load(args["WEIGHTS"], map_location=device)
model = weights['model']
_ = model.float().eval()

if torch.cuda.is_available():
    model.half().to(device)

# Run inference
if device.type != 'cpu':
    model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(
        next(model.parameters())))  # run once

args["MODEL"] = model


def detect(userdata, im0, image_mime):
    # Padded resize
    img = letterbox(im0, userdata["IMG_SIZE"], stride=userdata["STRIDE"], auto=True)[0]

    img_tensor = transforms.ToTensor()(img)
    img_tensor = torch.tensor(np.array([img_tensor.numpy()]))

    # Inference
    t1 = time.time()
    pred, _ = userdata['MODEL'](img_tensor)
    print(pred[..., 4].max())

    # Apply NMS
    pred = non_max_suppression_kpt(pred,
                                   userdata["CONF_THRESHOLD"],
                                   userdata["IOU_THRESHOLD"],
                                   nc=userdata['MODEL'].yaml['nc'],
                                   nkpt=userdata['MODEL'].yaml['nkpt'],
                                   kpt_label=True)
    with torch.no_grad():
        pred = output_to_keypoint(pred)

    t2 = time.time()

    annotated = im0.copy()

    gain = (img.shape[0] / im0.shape[0], img.shape[1] / im0.shape[1])

    detections_kpt = []

    for idx in range(pred.shape[0]):
        entry = {}
        scaled = []
        keypoints = []
        conf = []
        for counter, val in enumerate(pred[idx, 7:].T):
            if counter % 3 == 0:
                scaled.append(val / gain[1])
                xc = val / gain[1]
            elif counter % 3 == 1:
                scaled.append(val / gain[0])
                yc = val / gain[1]
                keypoints.append([xc, yc])
            else:
                scaled.append(val)
                conf.append(val)

        plot_skeleton_kpts(annotated, np.asarray(scaled), 3)  # pred[idx, 7:].T, 3)

        xmin, ymin = (pred[idx, 2] - pred[idx, 4] / 2) / gain[1], (pred[idx, 3] - pred[idx, 5] / 2) / gain[0]
        xmax, ymax = (pred[idx, 2] + pred[idx, 4] / 2) / gain[1], (pred[idx, 3] + pred[idx, 5] / 2) / gain[0]

        entry['bounding_box'] = {'xmin': xmin, 'ymin': ymin, 'width': xmax - xmin, 'height': ymax - ymin}
        entry['keypoint_coordinates'] = keypoints
        entry['confidence_scores'] = conf

        detections_kpt.append(entry)

        cv2.rectangle(annotated,
                      (int(xmin), int(ymin)),
                      (int(xmax), int(ymax)),
                      color=(255, 0, 0),
                      thickness=1,
                      lineType=cv2.LINE_AA)

    payload = {
        "model": userdata["MODEL_NAME"],
        "image": base64_encode(annotated),
        "inference_time": t2 - t1,
        "training_dataset": userdata["TRAINING_DATASET"],
        "detections": detections_kpt
    }

    msg = json.dumps(payload, cls=NumpyEncoder)
    client.publish(userdata['MQTT_OUT_0'], msg)
    payload["image"] = "%s... - truncated for logs" % payload["image"][0:32]
    logger.info(payload)


def on_message(c, userdata, msg):
    try:
        image_mime = str(msg.payload.decode("utf-8", "ignore"))
        _, image_base64 = image_mime.split(',', 1)
        image = Image.open(BytesIO(base64.b64decode(image_base64)))
        detect(userdata, im0=np.array(image), image_mime=image_mime)

    except Exception as e:
        logger.error('Error:', e)
        exit(1)


client = mqtt.Client(args['NAME'], clean_session=True, userdata=args)
client.on_connect = on_connect
client.on_message = on_message
client.connect(args['MQTT_SERVICE_HOST'], args['MQTT_SERVICE_PORT'])
# This runs the network code in a background thread and also handles reconnecting for you.
client.loop_forever()
