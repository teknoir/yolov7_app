import os
import sys
import cv2
import json
import torch
import base64
import logging
import numpy as np
from PIL import Image
from io import BytesIO
import paho.mqtt.client as mqtt
from utils.datasets import letterbox
from utils.torch_utils import select_device
from tracker.byte_tracker import BYTETracker
from models.experimental import attempt_load
from utils.general import non_max_suppression,check_img_size

APP_NAME = os.getenv('APP_NAME', 'object_detection_and_tracking_app_teknoir')

args = {
        'NAME': APP_NAME,
        'MQTT_SERVICE_HOST': os.getenv('MQTT_SERVICE_HOST', '127.0.0.1'),
        'MQTT_SERVICE_PORT': int(os.getenv('MQTT_SERVICE_PORT', '1883')),
        'MQTT_IN_0': os.getenv("MQTT_IN_0", "camera/images"),
        'MQTT_OUT_0': os.getenv("MQTT_OUT_0", f"{APP_NAME}/events"),
        
        'WEIGHTS': os.getenv("WEIGHTS", ""),
        'CLASS_NAMES': os.getenv("CLASS_NAMES", ""),
        'CLASSES_TO_DETECT': str(os.getenv("CLASSES_TO_DETECT","")),

        'CONF_THRESHOLD': float(os.getenv("CONF_THRESHOLD", 0.25)),
        'IMG_SIZE': int(os.getenv("CONF_THRESHOLD", 640)),
        'IOU_THRESHOLD': float(os.getenv("IOU_THRESHOLD", 0.45)),
        'AGNOSTIC_NMS': os.getenv("AGNOSTIC_NMS", ""),
        'AUGMENTED_INFERENCE':os.getenv("AUGMENTED_INFERENCE",""),
        
        'DEVICE': os.getenv("DEVICE", 'cpu'),  
        'MODEL_NAME': os.getenv("MODEL_NAME", "object_detection_and_tracking_model"),  
        'MQTT_VERSION': os.getenv("MQTT_VERSION", '3'),
        'MQTT_TRANSPORT': os.getenv("MQTT_TRANSPORT", 'tcp'),

        "TRACKER_THRESHOLD": float(os.getenv("TRACKER_THRESHOLD", 0.5)),
        "TRACKER_MATCH_THRESHOLD": float(os.getenv("TRACKER_MATCH_THRESOLD", 0.8)),
        "TRACKER_BUFFER": int(os.getenv("TRACKER_BUFFER", 30)),
        "TRACKER_FRAME_RATE": int(os.getenv("TRACKER_FRAME_RATE", 10)),
    }


#.....Initialization of Logger.....
logger = logging.getLogger(args['NAME'])
ch = logging.StreamHandler(sys.stdout)
ch.setLevel(logging.INFO)
formatter = logging.Formatter('[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)
logger.setLevel(logging.INFO)
logger.info("TΞꓘN01R")
logger.info("... App is Setting Up ...")
#..................................


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


#....Resetting the User Arguments ....
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
    if len(args["CLASSES_TO_DETECT"])==1:
        args["CLASSES_TO_DETECT"] = args["CLASS_NAMES"] .index(args["CLASSES_TO_DETECT"])
    else:
        args["CLASSES_TO_DETECT"] = args["CLASSES_TO_DETECT"].split(",")
        cls_ids = []
        for index,cls_name in enumerate(args["CLASSES_TO_DETECT"]):
            cls_id = args["CLASS_NAMES"].index(cls_name)
            cls_ids.append(cls_id)
        args["CLASSES_TO_DETECT"] = cls_ids
        del cls_ids
#........................


#.....Setting Up Device & Model......
logger.info("... Setting Up Device ...")
device=select_device(args["DEVICE"])
logger.info("... Using {} Device ...".format(device))
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
#...................................


'''
Object Tracking Class responsible to do object tracking on detections...
@__init__: this is responsible to Initialize tracker
@_new_tracker: this function will create new tracker object
@get_tracker: this function will check if tracker exist, If not then create new
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
            self._trackers[camera_id]=self._new_tracker()
        return self._trackers[camera_id]   
    
logger.info("... Tracking Class Object Initialized ...")
trackers=ObjectTrackers(args)


'''
Stacking Function is responsible to merge the list of images for multiprocessing at once
@Input: base64 images coming from camera, image size
@Output: refined model input and refine images list
'''
def stack_images(im0m_list):
    '''
    np.stack =  take list of images and stack them togather, output format is (5 (no of images), x, y, 3 (img channels))
    np.flip = take a stack and flip it, so that last index will become first index ([0,1,....,n])->([n,n-1,.....0])
    np.transpose =  take a list of flipped array and converts row to col and col to rows.
    '''
    model_input = np.transpose(np.flip(np.stack(im0m_list), 3), (0, 3, 1, 2)).astype(np.float32) / 255.0
    return model_input


'''
PreProcessing Function, which is responsible to preprocess data
@Input: Camera Data, which will include input data from camera and will include, base64 imgs and camera_ids
@Output: refine model_input, original images list and cameras ids list
'''
def preprocess(input_payload):
    im0m_list = []                              #...list to store resize images, for stacking, this will be removed after stacking operation..
    cm0_list = []                               #...list to store camera ids for output payload creation
    bs64_list = []                              #...list to store each camera image base64str for output payload creation  
    ts_list = []                                #...list to store each camera image timestamp for output payload creation
    im0_imgsz_list = []                         #...list to store each camera image size for output payload creation                


    #... Looop over each camera data
    for each_camera_data in input_payload: 

        '''@Research We will need to implementation condition, If images or someother data will be null
        in input, then code will display message instead of throwing error or crash or exit the app'''

        timestamp = each_camera_data["timestamp"]
        ts_list.append(timestamp)

        bs64_img = each_camera_data["images"] 
        bs64_list.append(bs64_img)
        
        cm0 = each_camera_data["camera_id"]
        cm0_list.append(cm0)
        
        #... Converting to Original Image
        _, image_base64 = bs64_img.split(',', 1)
        image = Image.open(BytesIO(base64.b64decode(image_base64)))
        image = cv2.cvtColor(np.array(image),cv2.COLOR_BGR2RGB)
        im0_imgsz_list.append(image.shape)

        #... Resized Image
        image = letterbox(image, imgsz, stride=stride)[0]   
        im0m_list.append(image)
    
    model_input = stack_images(im0m_list)

    #... This list will not be used anymore, better to delete
    del im0m_list

    return model_input,cm0_list,ts_list,im0_imgsz_list,bs64_list
    

'''
Detect Function, which is responsible to detect the objects
@Input: model input, which will be a list of tensors, to process multiple images at once
@Output: predictions data, which can be used for tracking and outpayload
'''
def detect(model_input):
    model_input = torch.tensor(model_input, device=device)
    with torch.no_grad():
        model_output = model(model_input)[0]
        #...Applied NMS to model output
        print(args["CLASSES_TO_DETECT"])
        detections = non_max_suppression(model_output, 
                                    args["CONF_THRESHOLD"],
                                    args["IOU_THRESHOLD"],
                                    classes=args["CLASSES_TO_DETECT"], 
                                    agnostic=args["AGNOSTIC_NMS"])
        #... This variable is not needed anymore, becuase now everywork will be based on pred variable
        del model_input,model_output    
    return detections


'''
track Function, which is responsible to track the objects
@Input: Predictions from detection model, and cameras_data list
@Output: tracked objects by the tracker, it can include trackid, track_bbox etc.
'''
def track(detections,cm0_list):
    tracked_objects_list = []
    for index, detection in enumerate(detections): 
        current_tracker = trackers.get_tracker(cm0_list[index])
        tracked_objects = current_tracker.update(torch.tensor(detection))
        del current_tracker
        tracked_objects_list.append(tracked_objects)
    return tracked_objects_list
    

'''
create_payload Function, which is responsible to create an output payload
@Input: Predictions from detection model, and cameras_data list
@Output: tracked objects by the tracker, it can include trackid, track_bbox etc.
'''
def create_payload(tracked_objects_list,ts_list,im0_imgsz_list,bs64_list):
    output_payload = []
    for index,tracked_object in enumerate(tracked_objects_list):
        result={}
        result["timestamp"]=ts_list[index]
  
        result["data"] = {}
        # result["data"]["image"]=bs64_list[index]
        result["data"]["image"]=""
        result["data"]["width"]=int(im0_imgsz_list[index][1])
        result["data"]["height"]=int(im0_imgsz_list[index][0])
        result["objects"]=[]
        for x1,x2,y1,y2,id,score in tracked_object:
            result["objects"].append({'trk_id':id,                                      
                                'x1':x1,
                                'y1':y1,
                                'x2':x2,
                                'y2':y2, 
                                'xcenter': (x1+x2)/2,
                                'ycenter': (y1+y2)/2,
                                'width': (x2-x1),
                                'height': (y2-y1),
                                'confidence': score, 
                                'area': x2*y2, 
                                }) 
        output_payload.append(result)
    return output_payload


'''
On_message Function, which is responsible to send and recieve messages from MQTT...
@Input: input payload that will comes from camera app
@Output: output payload that will include all the info of detection and tracking
'''
def on_message(c, userdata, msg):
    try:
        logger.info("... Message Recieved ...")
        recieved_message = msg.payload.decode("utf-8", "ignore")
        print(recieved_message)
        
        try:
            camera_data = json.loads(recieved_message)
        except json.JSONDecodeError as e:
            print("Error decoding JSON:", e)
            exit(1)

        input_payload = camera_data["payload"] 

        #... Preprocess the Input Payload ...
        model_input,cm0_list,ts_list,im0_imgsz_list,bs64_list = preprocess(input_payload) 

        #... Detect Objects in Images ... 
        detections = detect(model_input)

        #... Track Objects in Images ...
        tracked_objects_list = track(detections,cm0_list)

        #... Creation of Output Payload ...
        output_payload = create_payload(tracked_objects_list,ts_list,im0_imgsz_list,bs64_list)
        print(output_payload)
        
        #... Send Message through MQTT ...
        msg_to_send=json.dumps(output_payload, cls=NumpyEncoder)
        client.publish(userdata['MQTT_OUT_0'], msg_to_send)

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