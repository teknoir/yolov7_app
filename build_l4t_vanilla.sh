#!/usr/bin/env bash
set -eo pipefail

export SHORT_SHA=${SHORT_SHA:-"head"}
export BRANCH_NAME=${BRANCH_NAME:-"local"}
export PROJECT_ID=${PROJECT_ID:-"teknoir"}

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

build_and_push(){
  TAG=$1
  # Make sure the latest version of base image is local
  docker pull us-central1-docker.pkg.dev/${PROJECT_ID}/teknoir-ai/yolov7-base:${TAG}

  # Build and set values specific to this model
  docker buildx build \
    --build-arg=BASE_IMAGE=us-central1-docker.pkg.dev/${PROJECT_ID}/teknoir-ai/yolov7-base:${TAG} \
    --build-arg=MODEL_NAME=yolov7 \
    --build-arg=TRAINING_DATASET=coco \
    --build-arg=IMG_SIZE=640 \
    --build-arg=WEIGHTS_FILE=yolov7-tiny.pt \
    --build-arg=CLASS_NAMES_FILE=coco.names \
    --platform=linux/arm64 \
    --label "git-commit=${SHORT_SHA}" \
    --push \
    -t us-central1-docker.pkg.dev/${PROJECT_ID}/teknoir-ai/yolov7-vanilla:${TAG}-${BRANCH_NAME}-${SHORT_SHA} \
    -f ./vanilla.Dockerfile .
}

build_and_push l4tr34.1.1
build_and_push l4tr32.7.1
build_and_push l4tr32.7.2
