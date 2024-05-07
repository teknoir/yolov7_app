#!/usr/bin/env bash
set -eo pipefail
set -x

export BASE_IMAGE=${BASE_IMAGE:-"us-docker.pkg.dev/teknoir/gcr.io/yolov7-base:l4t-r36.2.0"}
export BRANCH_NAME=${BRANCH_NAME:-"main"}
export SHORT_SHA=${SHORT_SHA:-$(git rev-parse --short HEAD)}
export IMAGE_NAME=${IMAGE_NAME:-"us-docker.pkg.dev/teknoir/gcr.io/yolov7-vanilla"}
export TAG=${TAG:-"l4t-r36.2.0"}

# Get vanilla model and COCOA names file
download_cache(){
  FILE=$1
  URL=$2
  echo "Download ${FILE}"

  if [ -f "${FILE}" ]; then
    echo "${FILE} exists."
  else
    echo "Download ${FILE} from ${URL}"
    curl -fsSL --progress-bar -o ${FILE} ${URL} || {
      error "curl -fsSL --progress-bar -o ${FILE} ${URL}" "${FUNCNAME}" "${LINENO}"
      exit 1
    }
  fi
}
download_cache coco.names https://raw.githubusercontent.com/pjreddie/darknet/master/data/coco.names
download_cache yolov7-tiny.pt https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7-tiny.pt

docker buildx build \
  --builder mybuilder \
  --platform=linux/arm64/v8 \
  --push \
  --build-arg=MODEL_NAME=yolov7 \
  --build-arg=TRAINING_DATASET=coco \
  --build-arg=IMG_SIZE=640 \
  --build-arg=WEIGHTS_FILE=yolov7-tiny.pt \
  --build-arg=CLASS_NAMES_FILE=coco.names \
  --label "git-commit=${SHORT_SHA}" \
  --annotation 'index:teknoir.org/display-name="Yolo v7 Vanilla"' \
  --annotation 'index:teknoir.org/description="Yolo v7 for L4T trained on the COCO dataset."' \
  --annotation 'index:teknoir.org/framework=pythorch' \
  --annotation 'index:teknoir.org/image-type=model' \
  --annotation 'index:teknoir.org/model-type-description=yolov7' \
  --annotation 'index:teknoir.org/model-name=vanilla' \
  --annotation 'index:teknoir.org/model-dataset=coco' \
  --annotation 'index:teknoir.org/version=v1.0.0' \
  --annotation 'index:teknoir.org/model-feature=l4t' \
  --annotation 'index:github.com/project-slug=teknoir/yolov7-app' \
  --build-arg=BASE_IMAGE=${BASE_IMAGE} \
  --build-arg=BUILD_IMAGE=${BUILD_IMAGE} \
  --tag ${IMAGE_NAME}:${TAG}-${BRANCH_NAME}-${SHORT_SHA} \
  --file vanilla.Dockerfile \
  .

if [ ${BRANCH_NAME} == 'main' ]; then
  docker buildx build \
    --builder mybuilder \
    --platform=linux/arm64/v8 \
    --push \
    --build-arg=MODEL_NAME=yolov7 \
    --build-arg=TRAINING_DATASET=coco \
    --build-arg=IMG_SIZE=640 \
    --build-arg=WEIGHTS_FILE=yolov7-tiny.pt \
    --build-arg=CLASS_NAMES_FILE=coco.names \
    --label "git-commit=${SHORT_SHA}" \
    --annotation 'index:teknoir.org/display-name="Yolo v7 Vanilla"' \
    --annotation 'index:teknoir.org/description="Yolo v7 for L4T trained on the COCO dataset."' \
    --annotation 'index:teknoir.org/framework=pythorch' \
    --annotation 'index:teknoir.org/image-type=model' \
    --annotation 'index:teknoir.org/model-type-description=yolov7' \
    --annotation 'index:teknoir.org/model-name=vanilla' \
    --annotation 'index:teknoir.org/model-dataset=coco' \
    --annotation 'index:teknoir.org/version=v1.0.0' \
    --annotation 'index:teknoir.org/model-feature=l4t' \
    --annotation 'index:github.com/project-slug=teknoir/yolov7-app' \
    --build-arg=BASE_IMAGE=${BASE_IMAGE} \
    --build-arg=BUILD_IMAGE=${BUILD_IMAGE} \
    --tag ${IMAGE_NAME}:${TAG} \
    --file vanilla.Dockerfile \
    .
fi