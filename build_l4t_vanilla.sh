#!/usr/bin/env bash
set -eo pipefail

export SHORT_SHA=${SHORT_SHA:-"head"}
export BRANCH_NAME=${BRANCH_NAME:-"local"}
export PROJECT_ID=${PROJECT_ID:-"teknoir"}

# Get vanilla model and COCO names file
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
    --label "teknoir.org/app-name=yolov7-vanilla-${BRANCH_NAME}-${SHORT_SHA}" \
    --label "teknoir.org/app-type=model" \
    --label "teknoir.org/gpu=nvidia_l4t" \
    --label "teknoir.org/l4t-version=${TAG}" \
    --label "teknoir.org/model-type-description=yolov7-vanilla-object-detection" \
    --label "teknoir.org/model-name=yolov7-vanilla" \
    --label "teknoir.org/version=${BRANCH_NAME}-${SHORT_SHA}" \
    --label "teknoir.org/framework=pytorch" \
    --label "teknoir.org/memory-usage=low" \
    --label "teknoir.org/dataset-name=coco" \
    --label "teknoir.org/dataset-version=v1.0.0" \
    --label "teknoir.org/minimum-required-cpu=1" \
    --label "teknoir.org/minimum-required-ram=3GB" \
    --label "teknoir.org/minimum-required-gpu=0" \
    --push \
    -t us-central1-docker.pkg.dev/${PROJECT_ID}/teknoir-ai/yolov7-vanilla:${TAG}-${BRANCH_NAME}-${SHORT_SHA} \
    -f ./vanilla.Dockerfile .
}

build_and_push l4tr34.1.1
build_and_push l4tr32.7.1
