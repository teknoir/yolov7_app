# Teknoir YOLOv7 App

Build tool and notebooks to create the Yolov7 App for the Teknoir Platform

gcr.io/teknoir/yolov7:l4tr34.1.1

## Images
| Public images         |                                              Device |                               Alias tags |
|-----------------------|----------------------------------------------------:|-----------------------------------------:|
| gcr.io/teknoir/yolov7 | Generic<br/>amd64 / arm64<br/>(No GPU acceleration) |                        latest (manifest) |
| gcr.io/teknoir/yolov7 |      NVidia Jetson Nano<br/>NVidia Jetson Xavier NX | l4tr32.7.1<br/>l4tr32.7.2<br/>l4tr34.1.1 |
| gcr.io/teknoir/yolov7 |                           Raspberry Pi 4 (64bit OS) |                                      TBD |

## Example
In `vanilla.Dockerfile` and `build_vanilla.sh` there is a full example of a Yolov7 app for the Teknoir platform with the vanilla model trained on COCOA dataset.

| Public example images         |                                              Device |                               Alias tags |
|-------------------------------|----------------------------------------------------:|-----------------------------------------:|
| gcr.io/teknoir/yolov7-vanilla | Generic<br/>amd64 / arm64<br/>(No GPU acceleration) |                        latest (manifest) |
| gcr.io/teknoir/yolov7-vanilla |      NVidia Jetson Nano<br/>NVidia Jetson Xavier NX | l4tr32.7.1<br/>l4tr32.7.2<br/>l4tr34.1.1 |
| gcr.io/teknoir/yolov7-vanilla |                           Raspberry Pi 4 (64bit OS) |                                      TBD |

## Kubeflow Pipelines
In the `notebooks` dir there are 2 notebooks that create Pipeline Templates.

### Train Yolov7
Running the `train_yolov7.ipynb` notebook in the Teknoir Platform creates a pipeline that is prepared to take input from Labelstudio. See notebook for more information.

The notebook creates model artifacts that gets pushed to couldstorage and can be viewed in the Artifact Browser.


### Build a custom Yolov7
Running the `build_yolov7_app.ipynb` notebook in the Teknoir Platform creates a pipeline that is prepared to take input from the pipeline created by `train_yolov7.ipynb`. See notebook for more information. 

The notebook creates docker image artifact that gets pushed to the private Artifact Registry `us-central1-docker.pkg.dev/{project_id}/{namespace}/{image_name}`.
Where `{namespace}` is your projects name and `{image_name}` is specified in the Devstudio triggering the build.