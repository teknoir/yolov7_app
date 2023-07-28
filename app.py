import os
import sys
import cv2
import json
import time
import torch
import base64
import logging
import warnings
import numpy as np
from PIL import Image
from io import BytesIO
import paho.mqtt.client as mqtt
warnings.filterwarnings('ignore')
from tracker.byte_tracker import BYTETracker
from utils.torch_utils import select_device
from models.experimental import attempt_load
from utils.general import non_max_suppression,check_img_size
from memory_profiler import profile
APP_NAME = os.getenv('APP_NAME', 'object_tracking_app_teknoir')
args = {
        'NAME': APP_NAME,
        'MQTT_SERVICE_HOST': os.getenv('MQTT_SERVICE_HOST', '127.0.0.1'),
        'MQTT_SERVICE_PORT': int(os.getenv('MQTT_SERVICE_PORT', '1883')),
        'MQTT_IN_0': os.getenv("MQTT_IN_0", "camera/images"),
        'MQTT_OUT_0': os.getenv("MQTT_OUT_0", f"{APP_NAME}/events"),
        'WEIGHTS': os.getenv("WEIGHTS", "E:\\Weights\\yolov7-tiny.pt"),
        'CLASS_NAMES': os.getenv("CLASS_NAMES", "classes.names"),
        'CLASSES_TO_DET': os.getenv("CLASSES_TO_DET",[0,2]),
        'CONF_THRESHOLD': float(os.getenv("CONF_THRESHOLD", 0.25)),
        'IMG_SIZE': int(os.getenv("CONF_THRESHOLD", 640)),
        'IOU_THRESHOLD': float(os.getenv("IOU_THRESHOLD", 0.45)),
        "TRACKER_THRESHOLD": float(os.getenv("TRACKER_THRESHOLD", 0.5)),
        "TRACKER_MATCH_THRESHOLD": float(os.getenv("TRACKER_MATCH_THRESOLD", 0.8)),
        "TRACKER_BUFFER": int(os.getenv("TRACKER_BUFFER", 30)),
        "TRACKER_FRAME_RATE": int(os.getenv("TRACKER_FRAME_RATE", 10)),
        'DEVICE': os.getenv("DEVICE", 'cpu'),  
        'MODEL_NAME': os.getenv("MODEL_NAME", "object_tracking_app_teknoir"),  
        'MQTT_VERSION': os.getenv("MQTT_VERSION", '3'),
        'MQTT_TRANSPORT': os.getenv("MQTT_TRANSPORT", 'tcp'),
    }


#... Initialization of Logger ...
logger = logging.getLogger(args['NAME'])
ch = logging.StreamHandler(sys.stdout)
ch.setLevel(logging.INFO)
formatter = logging.Formatter('[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)
logger.setLevel(logging.INFO)
logger.info("TΞꓘN01R")
logger.info("... App is Setting Up ...")


#... Classes Names Configuration ....
if args["CLASS_NAMES"] != "":
    class_names = []
    with open(args["CLASS_NAMES"],"r",encoding='utf-8') as names_file:
        for line in names_file:
            if line != "" and line != "\n":
                class_names.append(line.strip())
    args["CLASS_NAMES"] = class_names
else:
    logger.info("You must specify 'CLASS_NAMES'")
    logger.info("App Exit!")
    sys.exit(1)
logger.info("... Classes Names Configured ...")


def error_str(rc):
    return '{}: {}'.format(rc, mqtt.error_string(rc))

def on_connect_v3(client, _userdata, _flags, rc):
    print('Connected to MQTT broker {}'.format(error_str(rc)))
    if rc == 0:
        client.subscribe(args['MQTT_IN_0'], qos=0)
        
def on_connect_v5(client, _userdata, _flags, rc, _props):
    print('Connected to MQTT broker {}'.format(error_str(rc)))
    if rc == 0:
        client.subscribe(args['MQTT_IN_0'], qos=0)


#... Setting Up Device & Model
logger.info("... Setting Up Device ...")
device=select_device(args["DEVICE"])
logger.info("... Using {} Device ...".format(device))
half=device.type != 'cpu'
model = attempt_load(args["WEIGHTS"], map_location=device)
logger.info("... Model Initialized ...")
names = model.module.names if hasattr(model, 'module') else model.names
if half:
    model.half()


#... Check Image Size ...
stride = int(model.stride.max())
imgsz = check_img_size(args["IMG_SIZE"], s=stride)

class NumpyEncoder(json.JSONEncoder):
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


#...Object Tracking Class responsible to do object tracking on detections...
'''
__init__: this is responsible to Initialize tracker
_new_tracker: this function will create new tracker object
get_tracker: this function will check if tracker exist, If not then create new
otherwise return old based on camera id
'''
class ObjectTrackers:
    def __init__(self, args):
        self._trackers={}
        self.threshold=args["TRACKER_THRESHOLD"]
        self.match_threshold=args["TRACKER_MATCH_THRESHOLD"]
        self.buffer=args["TRACKER_BUFFER"]
        self.frame_rate=args["TRACKER_FRAME_RATE"]

    def _new_tracker(self):
        return BYTETracker(track_buffer=self.buffer,
                            frame_rate=self.frame_rate,
                            track_thresh=self.threshold,
                            match_thresh=self.match_threshold)   
        
    def get_tracker(self, camera_id):
        if camera_id not in self._trackers:
            logger.info("... New Object Tracker Initialized ...")
            self._trackers[camera_id]=self._new_tracker()
        return self._trackers[camera_id]   
    
logger.info("... Tracking Class Object Initialized ...")
trackers=ObjectTrackers(args)


#...Stacking Function is responsible to merge the list of images for multiprocessing at once...
'''
Input: base64 images coming from camera, input width and height of model
Output: refined model input and refine images list
'''
def stack_images(base64_images_list,model_imh,model_imw):
    im0_list = []    
    im0_m_list = []
    for im0_ind in range(0,len(base64_images_list)):
        _, image_base64 = base64_images_list[im0_ind].split(',', 1)
        image = Image.open(BytesIO(base64.b64decode(image_base64)))
        im0_list.append(np.array(image))
        im0_res =cv2.resize(np.array(image), (model_imw,model_imh))
        im0_res = cv2.cvtColor(im0_res,cv2.COLOR_BGR2RGB)        
        im0_m_list.append(im0_res)
    model_input = np.transpose(np.flip(np.stack(im0_m_list), 3), (0, 3, 1, 2)).astype(np.float32) / 255.0
    del im0_res,im0_m_list
    return model_input,im0_list


#...PreProcessing Function, which is responsible to preprocess data....
'''
Input: Camera Data, which will include input data from camera and will include, base64 imgs and camera_ids
Output: refine model_input, original images list and cameras ids list
'''
def preprocess(camera_data):
    if "images" not in camera_data:
        logging.info("Input not include images data!")
        logging.info("App Exit!")
        sys.exit(1)
    elif "camera_id" not in camera_data:
        logging.info("Input not include camera_id data!")
        logging.info("App Exit!")
        sys.exit(1)
    else:
        base64_images_list = camera_data["images"] 
        cm0_list = camera_data["camera_id"]
        model_input,im0_list = stack_images(base64_images_list,imgsz,imgsz)
        return model_input,im0_list,cm0_list,base64_images_list


#...Detect Function, which is responsible to detect the objects....
'''
Input: model input, which will be a list of tensors, to process multiple images at once
Output: predictions data, which can be used for tracking and outpayload
'''
def detect(model_input):
    model_input = torch.tensor(model_input, device=device)
    with torch.no_grad():
        model_output = model(model_input)[0]
        pred = non_max_suppression(model_output, 
                                    args["CONF_THRESHOLD"],
                                    args["IOU_THRESHOLD"],
                                    classes=args["CLASSES_TO_DET"], 
                                    agnostic=False)
        del model_input,
        del model_output
    return pred


#...track_and_outpayload Function, which is responsible to track the objects and payload creation....
'''
Input: Predictions from detection model and cameras_data list
Output: tracked objects by the tracker, which will include all output payload and generate out payload
'''
def track_and_outpayload(pred,base64_images_list,im0_list,cm0_list):
    out_payload = {"model": {"name": "yolov7","version": "0.1","id": "stack_0.1",},
                    "results": [{"timestamp": [],"camera": [],"data": [],"objects": []}]}
    for ind, det in enumerate(pred):
        if len(det):
            new_object = []
            im0_list[ind] = cv2.cvtColor(im0_list[ind],cv2.COLOR_BGR2RGB) 
            current_tracker = trackers.get_tracker(cm0_list[ind])
            tracked_objects = current_tracker.update(torch.tensor(det))
            for tracked_object in tracked_objects: 
                x1 = tracked_object[0]                                              
                y1 = tracked_object[1]                                              
                x2 = tracked_object[2]                                              
                y2 = tracked_object[3]   
                x_center = (x1 + x2) / 2
                y_center = (y1 + y2) / 2
                track_id = tracked_object[4]                                       
                class_index = int(tracked_object[5])                            
                score = tracked_object[6]  
                
                new_object.append({'trk_id':track_id,                                      
                                    'x1':x1,
                                    'y1':y1,
                                    'x2':x2,
                                    'y2':y2, 
                                    'xcenter': x_center,
                                    'ycenter': y_center,
                                    'width': (x2-x1),
                                    'height': (y2-y1),
                                    'confidence': score, 
                                    'area': x2*y2, 
                                    'label': args["CLASS_NAMES"][class_index],
                                }) 
                xyxy_rescaled = [
                                    int(tracked_object[0] * im0_list[ind].shape[1] / args["IMG_SIZE"]),
                                    int(tracked_object[1] * im0_list[ind].shape[0] / args["IMG_SIZE"]),
                                    int(tracked_object[2] * im0_list[ind].shape[1] / args["IMG_SIZE"]),
                                    int(tracked_object[3] * im0_list[ind].shape[0] / args["IMG_SIZE"]),
                                ]

            new_camera={"id": cm0_list[ind], 
                        "name": "name_{}".format(cm0_list[ind])}
            
            new_data=   {"image": base64_images_list[ind],
                         "width": im0_list[ind].shape[1], 
                         "height": im0_list[ind].shape[0]}
            
            out_payload["results"][0]["timestamp"].append(time.time())
            out_payload["results"][0]["camera"].append(new_camera)
            out_payload["results"][0]["data"].append(new_data)
            out_payload["results"][0]["objects"].append(new_object)

        else:
            logger.info("... No Det...Skipping the Tracking ...!")
    return out_payload


#...On_message Function, which is responsible to send and recieve messages from MQTT...
'''
Input: input payload that will comes from camera app
Output: output payload that will include all the info of detection and tracking
'''
@profile
def on_message(c, userdata, msg):
    try:
        logger.info("... Message Recieved ...")
        input_payload = str(msg.payload.decode("utf-8", "ignore"))                            
        camera_data = json.loads(input_payload)  
        
        model_input,im0_list,cm0_list,base64_images_list = preprocess(camera_data)  
        pred = detect(model_input)
        output_payload = track_and_outpayload(pred,base64_images_list,im0_list,cm0_list)

        out_msg = json.dumps(output_payload, cls=NumpyEncoder)
        client.publish(userdata['MQTT_OUT_0'], out_msg)
        
        
        for result in output_payload["results"]:
            for obj in result["data"]:
                if "image" in obj:
                    obj["image"] = obj["image"][:32]
        
        print(output_payload)
        del input_payload,camera_data,model_input,im0_list,cm0_list,pred
        del output_payload
    
    except Exception as e:
        print('Error:', e)
        exit(1)
     

if args['MQTT_VERSION'] == '5':
    client = mqtt.Client(client_id=args['NAME'],
                         transport=args['MQTT_TRANSPORT'],
                         protocol=mqtt.MQTTv5,
                         userdata=args)
    client.reconnect_delay_set(min_delay=1, max_delay=120)
    client.on_connect = on_connect_v5
    client.on_message = on_message
    client.connect(args['MQTT_SERVICE_HOST'],port=args['MQTT_SERVICE_PORT'],
                   clean_start=mqtt.MQTT_CLEAN_START_FIRST_ONLY,keepalive=60)

if args['MQTT_VERSION'] == '3':
    client = mqtt.Client(client_id=args['NAME'],transport=args['MQTT_TRANSPORT'],
                         protocol=mqtt.MQTTv311,userdata=args,clean_session=True)
    client.reconnect_delay_set(min_delay=1, max_delay=120)
    client.on_connect = on_connect_v3
    client.on_message = on_message
    client.connect(args['MQTT_SERVICE_HOST'], port=args['MQTT_SERVICE_PORT'], keepalive=60)

client.loop_forever()