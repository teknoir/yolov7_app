# Teknoir YOLOv7 App

Build tool and notebooks to create the Yolov7 App for the Teknoir Platform

## Images
| Public images         |                                     Device |                                     Alias tags |
|-----------------------|-------------------------------------------:|-----------------------------------------------:|
| gcr.io/teknoir/yolov7 |        Generic<br/>amd64 / arm64<br/>(CPU) |          latest (manifest)<br/>amd64<br/>arm64 |
| gcr.io/teknoir/yolov7 | Generic<br/>amd64 / arm64<br/>(Nvidia GPU) | nv-latest (manifest)<br/>nv-amd64<br/>nv-arm64 |
| gcr.io/teknoir/yolov7 |  NVidia L4T<br/>Jetson Nano/Xavier NX/Orin |       l4tr32.7.1<br/>l4tr32.7.2<br/>l4tr34.1.1 |
| gcr.io/teknoir/yolov7 |                  Raspberry Pi 4 (64bit OS) |                                            TBD |

## Example
In `vanilla.Dockerfile` and `build_vanilla.sh` there is a full example of a Yolov7 app for the Teknoir platform with the vanilla model trained on COCOA dataset.

| Public example images         |                                     Device |                               Alias tags |
|-------------------------------|-------------------------------------------:|-----------------------------------------:|
| gcr.io/teknoir/yolov7-vanilla |        Generic<br/>amd64 / arm64<br/>(CPU) |                        latest (manifest) |
| gcr.io/teknoir/yolov7-vanilla | Generic<br/>amd64 / arm64<br/>(Nvidia GPU) |                     nv-latest (manifest) |
| gcr.io/teknoir/yolov7-vanilla |  NVidia L4T<br/>Jetson Nano/Xavier NX/Orin | l4tr32.7.1<br/>l4tr32.7.2<br/>l4tr34.1.1 |
| gcr.io/teknoir/yolov7-vanilla |                  Raspberry Pi 4 (64bit OS) |                                      TBD |

## Kubeflow Pipelines
In the `notebooks` dir there are 2 notebooks that create Pipeline Templates.

### Train Yolov7
Running the `train_yolov7.ipynb` notebook in the Teknoir Platform creates a pipeline that is prepared to take input from Labelstudio. See notebook for more information.

The notebook creates model artifacts that gets pushed to couldstorage and can be viewed in the Artifact Browser.

### Build a custom Yolov7
Running the `build_yolov7_app.ipynb` notebook in the Teknoir Platform creates a pipeline that is prepared to take input from the pipeline created by `train_yolov7.ipynb`. See notebook for more information. 

The notebook creates docker image artifact that gets pushed to the private Artifact Registry `us-central1-docker.pkg.dev/{project_id}/{namespace}/{image_name}`.
Where `{namespace}` is your projects name and `{image_name}` is specified in the Devstudio triggering the build.

## Teknoir Devstudio
When the pipelines are created there is an example flow in the `flows` folder that can be imported in a Devstudio.

## Local Testing

Assuming you have a MQTT broker on your local network and a device that is emitting MQTT messages (possibly the same device), the following can be used to test this repository:

1. Download a model artifact from GCS (like ppe-bbox-clean-20220821000000146).

2. Run eclipse-mosquitto (in separate terminal window)

`docker pull eclipse-mosquitto`

`docker run -it --name mosquitto -p 1883:1883 eclipse-mosquitto`

3. Execute code below with configuration modifications to reflect your system, data, etc. 

`MQTT_IN_0="camera/images" MQTT_SERVICE_HOST=192.168.68.104 MQTT_SERVICE_PORT=31883 WEIGHTS=weights/best_weights.pt IMG_SIZE=640 CLASS_NAMES=ppe-bbox-clean-20220821000000146/dataset/object.names python3 app.py`

