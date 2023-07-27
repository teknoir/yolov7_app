import numpy as np
from PIL import Image
from io import BytesIO
import os, sys,cv2,json
import random,time,base64,torch
import paho.mqtt.client as mqtt
from models.experimental import attempt_load
from tracker.byte_tracker import BYTETracker
from utils.general import non_max_suppression,check_img_size

APP_NAME = os.getenv('APP_NAME', 'stacking_app_test')

args = {
        'NAME': APP_NAME,
        'MQTT_SERVICE_HOST': os.getenv('MQTT_SERVICE_HOST', '127.0.0.1'),
        'MQTT_SERVICE_PORT': int(os.getenv('MQTT_SERVICE_PORT', '1883')),
        'MQTT_IN_0': os.getenv("MQTT_IN_0", "camera/images"),
        'MQTT_OUT_0': os.getenv("MQTT_OUT_0", f"{APP_NAME}/events"),
        'WEIGHTS': os.getenv("WEIGHTS", "yolov7-tiny.pt"),
        'STREAMS': int(os.getenv("STREAMS",4)),
        'CLASS_NAMES': os.getenv("CLASS_NAMES", "classes.names"),
        'CLASSES_TO_DET': os.getenv("CLASSES_TO_DET",[0,2]),
        'CONF_THRESHOLD': float(os.getenv("CONF_THRESHOLD", 0.25)),
        'IMG_SIZE': int(os.getenv("CONF_THRESHOLD", 640)),
        'IOU_THRESHOLD': float(os.getenv("IOU_THRESHOLD", 0.45)),
        'DEVICE': os.getenv("DEVICE", 'cpu'),  
        'MODEL_NAME': os.getenv("MODEL_NAME", "object_detection_and_tracking_app_teknoir"),  
        'MQTT_VERSION': os.getenv("MQTT_VERSION", '3'),
        'MQTT_TRANSPORT': os.getenv("MQTT_TRANSPORT", 'tcp'),
    }

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

print("TΞꓘN01R")
print("... App is Setting Up ...")
print("... Detection Model initialization...")

device = "cpu"
model = attempt_load(args["WEIGHTS"], map_location=device)
stride = int(model.stride.max())
imgsz = check_img_size(args["IMG_SIZE"], s=stride)
names = model.module.names if hasattr(model, 'module') else model.names
colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

print("... Classes Name Configuring ...")
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
print("... Classes Name Configured ...")

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

tracker_init_list = []
print("... Tracking Model Initialization ...")
for ind in range(args["STREAMS"]):
    tracker_init = BYTETracker(0.5,0.8,30,30)
    tracker_init_list.append(tracker_init)

'''
Function to convert group of Images to stack
'''
def stack_images(im0_list,model_imh,model_imw):
    im0 = []    
    im0_m = []
    for im0_ind in range(0,len(im0_list)):
        _, image_base64 = im0_list[im0_ind].split(',', 1)
        image = Image.open(BytesIO(base64.b64decode(image_base64)))
        im0.append(np.array(image))
        im0_res =cv2.resize(np.array(image), (model_imw,model_imh))
        im0_res = cv2.cvtColor(im0_res,cv2.COLOR_BGR2RGB)        
        im0_m.append(im0_res)
    model_input = np.transpose(np.flip(np.stack(im0_m), 3), (0, 3, 1, 2)).astype(np.float32) / 255.0
    del im0_m
    return model_input,im0
#............................................


'''
Detection Function Responsible for Handling stack of Images
'''
def detect(im0_list):
    img_size = args["IMG_SIZE"]
    out_payload = {"model": 
                    {
                        "name": "yolov7",
                        "version": "0.1",
                            "id": "stack_0.1",
                    },
                    "results": 
                    [
                        {
                            "timestamp": [],
                            "camera": [],
                            "data": [],
                            "objects": []
                         }
                    ]
                }
    
    model_input,im0 = stack_images(im0_list,img_size,img_size)
    model_input = torch.tensor(model_input, device=device)
    
    with torch.no_grad():
        model_output = model(model_input)[0]
        pred = non_max_suppression(model_output, 
                                    args["CONF_THRESHOLD"],
                                    args["IOU_THRESHOLD"],
                                    classes=args["CLASSES_TO_DET"], 
                                    agnostic=False)
        del img_size
        del model_input,
        del model_output
        
        for i, det in enumerate(pred):
            if len(det):
                new_object = [] 
                tracked_objects = tracker_init_list[i].update(torch.tensor(det), im0)
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
                    new_object.append({
                                        'trk_id':track_id,                                      
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
                new_camera={
                                "id": i, 
                                "name": "name_{}".format(i),
                            }
                                
                new_data=   {
                                "image": im0_list[i],
                                "width": im0[i].shape[1], 
                                "height": im0[i].shape[0],
                            }
                out_payload["results"][0]["timestamp"].append(time.time())
                out_payload["results"][0]["camera"].append(new_camera)
                out_payload["results"][0]["data"].append(new_data)
                out_payload["results"][0]["objects"].append(new_object)

    del im0,new_camera,new_data,pred
    return out_payload
#..........................................................


'''
Function use to handle messages from MQTT In
'''
def on_message(c, userdata, msg):
    try:
        message = str(msg.payload.decode("utf-8", "ignore"))                            
        data_recieved = json.loads(message)                                            

        images_list = data_recieved["images"]                           
        out_payload = detect(images_list)
    
        msg = json.dumps(out_payload, cls=NumpyEncoder)
        client.publish(userdata['MQTT_OUT_0'], msg)
        
        for result in out_payload["results"]:
            for obj in result["data"]:
                if "image" in obj:
                    obj["image"] = obj["image"][:32]
        
        print(out_payload)
        del message,data_recieved,images_list,msg,out_payload
    
    except Exception as e:
        print('Error:', e)
        exit(1)
#..........................................       


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