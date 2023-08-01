import os
import sys
import cv2
import json
import torch
import logging
import numpy as np
from PIL import Image
from io import BytesIO
import paho.mqtt.client as mqtt
from utils.torch_utils import select_device
from tracker.byte_tracker import BYTETracker
from models.experimental import attempt_load
from utils.general import non_max_suppression,check_img_size

APP_NAME = os.getenv('APP_NAME', 'object_detection_and_tracking_app_teknoir')

args = {
        'NAME': APP_NAME,
        
        'MQTT_IN_0': os.getenv("MQTT_IN_0", "camera/images"),
        'MQTT_OUT_0': os.getenv("MQTT_OUT_0", f"{APP_NAME}/events"),
        'MQTT_VERSION': os.getenv("MQTT_VERSION", '3'),
        'MQTT_TRANSPORT': os.getenv("MQTT_TRANSPORT", 'tcp'),
        'MQTT_SERVICE_HOST': os.getenv('MQTT_SERVICE_HOST', '127.0.0.1'),
        'MQTT_SERVICE_PORT': int(os.getenv('MQTT_SERVICE_PORT', '1883')),

        'DEVICE': os.getenv("DEVICE", 'cpu'),
        
        'AGNOSTIC_NMS': os.getenv("AGNOSTIC_NMS", ""),
        'IMG_SIZE': int(os.getenv("CONF_THRESHOLD", 640)),
        'CLASS_NAMES': os.getenv("CLASS_NAMES", "classes.names"),
        'IOU_THRESHOLD': float(os.getenv("IOU_THRESHOLD", 0.45)),
        'AUGMENTED_INFERENCE':os.getenv("AUGMENTED_INFERENCE",""),
        'CONF_THRESHOLD': float(os.getenv("CONF_THRESHOLD", 0.25)),
        'WEIGHTS': os.getenv("WEIGHTS", "E:\Weights\yolov7-tiny.pt"),
        'CLASSES_TO_DETECT': str(os.getenv("CLASSES_TO_DETECT","person,car")),
        'MODEL_NAME': os.getenv("MODEL_NAME", "object_detection_and_tracking_model"),  
        
        "TRACKER_THRESHOLD": float(os.getenv("TRACKER_THRESHOLD", 0.5)),
        "TRACKER_MATCH_THRESHOLD": float(os.getenv("TRACKER_MATCH_THRESOLD", 0.8)),
        "TRACKER_BUFFER": int(os.getenv("TRACKER_BUFFER", 30)),
        "TRACKER_FRAME_RATE": int(os.getenv("TRACKER_FRAME_RATE", 10)),
    }


# Initialization of Logger
logger = logging.getLogger(args['NAME'])
ch = logging.StreamHandler(sys.stdout)
ch.setLevel(logging.INFO)
formatter = logging.Formatter('[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)
logger.setLevel(logging.INFO)
logger.info("TΞꓘN01R")
logger.info("... App is Setting Up ...")


# MQTT Error and success handling functions
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

if args["CLASSES_TO_DETECT"] == "":
    args["CLASSES_TO_DETECT"] = None
else:
    cls_to_detect = args["CLASSES_TO_DETECT"]
    if len(cls_to_detect)==1:
        cls_to_detect = args["CLASS_NAMES"].index(cls_to_detect)
    else:
        cls_to_detect = cls_to_detect.split(",")
        cls_ids = []
        for index,cls_name in enumerate(cls_to_detect):
            cls_id = args["CLASS_NAMES"].index(cls_name)
            cls_ids.append(cls_id)
        args["CLASSES_TO_DETECT"] = cls_ids
        del cls_ids,cls_to_detect


# Setting Up Device & Model
logger.info("Setting Up Device ...")
device=select_device(args["DEVICE"])
logger.info("Using {} Device ...".format(device))
half=device.type != 'cpu'
model=attempt_load(args["WEIGHTS"], map_location=args["DEVICE"])
logger.info("... Model Initialized ...")
stride=int(model.stride.max())
imgsz=check_img_size(args["IMG_SIZE"], s=stride)
if isinstance(imgsz, (list, tuple)):
    assert len(imgsz) == 2
    imgsz[0]=check_img_size(imgsz[0], s=stride)
    imgsz[1]=check_img_size(imgsz[1], s=stride)
else:
    imgsz=check_img_size(imgsz, s=stride)
names=model.module.names if hasattr(model, 'module') else model.names
if half:
    model.half()
stride = int(model.stride.max())


class NumpyEncoder(json.JSONEncoder):
    """NumpyEncoder Class responsible to encode data into Json for MQTT out...
    Functions:
        default: the function will encode output paylaod to json
    """
    def default(self, obj):
        """This function will convert the output payload to Json Encoded data
        args:
            output payload object
        Returns: 
            output json encoded data
        """
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


class ObjectTrackers:
    """Object Tracking Class responsible to do object tracking on detections...
    Functions:
        __init__: this is responsible to Initialize tracker
        _new_tracker: this function will create new tracker object
        get_tracker: this function will check if tracker exist, If not then create new otherwise 
        return old based on camera id
    """

    def __init__(self, args):
        """class initializer
        args:
            args
        Returns
            None 
        """
        self._trackers={}
        self.threshold=args["TRACKER_THRESHOLD"]
        self.match_threshold=args["TRACKER_MATCH_THRESHOLD"]
        self.buffer=args["TRACKER_BUFFER"]
        self.frame_rate=args["TRACKER_FRAME_RATE"]
    
    def _new_tracker(self):
        """function to create new tracker
        args:
            None
        Returns:
            object of tracker
        """
        return BYTETracker(track_buffer=self.buffer,
                            frame_rate=self.frame_rate,
                            track_thresh=self.threshold,
                            match_thresh=self.match_threshold)   
    
    def get_tracker(self, camera_id):
        """function to get or create tracker
        args:
            camera_id
        Returns:
            object of tracker
        """
        if camera_id not in self._trackers:
            self._trackers[camera_id]=self._new_tracker()
        return self._trackers[camera_id]   

logger.info("... Tracking Class Object Initialized ...")
trackers=ObjectTrackers(args)


def stack_images(im0m_list):
    """Stacking function is responsible to merge the list of images for multiprocessing at once
    Args: 
        base64 images coming from camera, image size
    Returns: 
        refined model input and refine images list
    """
    
    # np.stack =  take list of images and stack them togather, output format is (5 (no of images), x, y, 3 (img channels))
    # np.flip = take a stack and flip it, so that last index will become first index ([0,1,....,n])->([n,n-1,.....0])
    # np.transpose =  take a list of flipped array and converts row to col and col to rows.
    
    model_input = np.transpose(np.flip(np.stack(im0m_list), 3), (0, 3, 1, 2)).astype(np.float32) / 255.0
    return model_input


def detect(model_input):
    """Detect function, which is responsible to detect the objects
    Args: 
        model input, which will be a list of tensors, to process multiple images at once
    Returns: 
        predictions data, which can be used for tracking and outpayload
    """
    # convert the input of model to torch tensors
    model_input = torch.tensor(model_input, device=device)
    # get detection from model
    model_output = model(model_input)[0]
    # Apply Non Maximum Suppression (NMS)
    detections = non_max_suppression(model_output,
                                    args["CONF_THRESHOLD"],
                                    args["IOU_THRESHOLD"],
                                    args["CLASSES_TO_DETECT"],
                                    args["AGNOSTIC_NMS"])
    print("CLASSES TO DETECT : ",args["CLASSES_TO_DETECT"])
    return detections


def track(detections,cm0_list):
    """track function, which is responsible to track the objects
    Args: 
        Predictions from detection model, and cameras_data list
    Returns: 
        tracked objects by the tracker, it can include trackid, track_bbox etc.
    """
    # list to store all tracked objects
    tracked_objects_list = []
    # loop over detections
    for index, detection in enumerate(detections):
        current_tracker = trackers.get_tracker(cm0_list[index])
        tracked_objects = current_tracker.update(torch.tensor(detection))
        tracked_objects_list.append(tracked_objects)
    return tracked_objects_list
    

'''
create_payload Function, which is responsible to create an output payload
@Input: Predictions from detection model, and cameras_data list
@Output: tracked objects by the tracker, it can include trackid, track_bbox etc.
'''
# def create_payload(tracked_objects_list,ts_list,im0_imgsz_list):
#     output_payload = []
#     for index,tracked_object in enumerate(tracked_objects_list):
#         result={}
#         result["timestamp"]=ts_list[index]
#         result["data"] = {}
#         result["data"]["image"]=bs64_list[index]
#         result["data"]["image"]=""
#         result["data"]["width"]=int(im0_imgsz_list[index][1])
#         result["data"]["height"]=int(im0_imgsz_list[index][0])
#         result["objects"]=[]

        # for xywh,id,score in tracked_object:
        #     result["objects"].append({'trk_id':id,                                      
        #                                 'x1':xywh[0],
        #                                 'y1':xywh[1],
        #                                 'width': xywh[2],
        #                                 'height': xywh[3],
        #                                 'confidence': score, 
        #                             }) 
    #     output_payload.append(result)
    # return output_payload


def on_message(c, userdata, msg):
    """On_message function, which is responsible to send and recieve messages from MQTT...
    Args: 
        input payload that will comes from camera app
    Returns: 
        output payload that will include all the info of detection and tracking
    """
    try:
        import base64
        logger.info("... Message Recieved ...")
        # decode the message
        recieved_message = msg.payload.decode("utf-8", "ignore")
        # try/except to avoid errors
        try:
            # load json str
            camera_data = json.loads(recieved_message)
        except json.JSONDecodeError as e:
            print("Error decoding JSON:", e)
            exit(1)

        # PreProcessing of input payload
        input_payload = camera_data["payload"]
        cm0_list = []   # list to store camera id's
        ts_list = []    # list to store
        im0m_list = []  # list to store im0_resized
        
        # loop over the input data
        for input in input_payload:
            # Append camera id to cameras list
            cm0_list.append(input["camera_id"])
            # Append timestamp to timestamps list
            ts_list.append(input["timestamp"])
            
            # store base64 image string
            bs64_img = input["image"]    
            # preprocess of base64 string     
            _, image_base64 = bs64_img.split(',', 1)
            # Decode the base64 string to byte64 array and read image
            image = Image.open(BytesIO(base64.b64decode(image_base64)))
            # Convert the BGR image to RGB
            image = cv2.cvtColor(np.array(image),cv2.COLOR_BGR2RGB)
            # Resize an image to model size
            image =cv2.resize(image,(args["IMG_SIZE"],args["IMG_SIZE"]))
            # Append resized image to im0_m list  
            im0m_list.append(image)
        # Stack Images 
        model_input = stack_images(im0m_list)
       
        # Detect Objects in Images 
        detections = detect(model_input)
        
        # Track Objects in Images
        tracked_objects_list = track(detections,cm0_list)
        print(tracked_objects_list)
        #... Creation of Output Payload ...
        # output_payload = create_payload(tracked_objects_list,ts_list,im0m_list)
        # print(output_payload)
        print("Completed ...")
        
        #... Send Message through MQTT ...
        # msg_to_send=json.dumps(output_payload, cls=NumpyEncoder)
        # client.publish(userdata['MQTT_OUT_0'], msg_to_send)

    except Exception as e:
        print('Error:', e)
        exit(1)


# MQTT Modules, we dont need to change anything inside
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

# used to loop on video file until the node disconnected.
client.loop_forever()