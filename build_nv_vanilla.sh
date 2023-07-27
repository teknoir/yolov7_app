#!/usr/bin/env bash
set -eo pipefail

export SHORT_SHA=${SHORT_SHA:-"head"}
export BRANCH_NAME=${BRANCH_NAME:-"local"}
export PROJECT_ID=${PROJECT_ID:-"teknoir"}

build_and_push(){
  BASE_TAG=$1
  TAG=$2
  ARCH=$3
  # Make sure the latest version of base image is local
  docker pull --platform=linux/${ARCH} gcr.io/teknoir/yolov7:${BASE_TAG}

  # Build and set values specific to this model
  docker buildx build \
    --build-arg=BASE_IMAGE=gcr.io/teknoir/yolov7:${BASE_TAG} \
    --build-arg=MODEL_NAME=yolov7-vanilla \
    --build-arg=TRAINING_DATASET=cocoa \
    --build-arg=IMG_SIZE=640 \
    --build-arg=WEIGHTS_FILE=yolov7.pt \
    --build-arg=CLASS_NAMES_FILE=coco.names \
    --platform=linux/${ARCH} \
    --label "git-commit=${SHORT_SHA}" \
    --load \
    -t gcr.io/${PROJECT_ID}/yolov7-vanilla:${TAG} \
    -f ./vanilla.Dockerfile .

  docker push gcr.io/${PROJECT_ID}/yolov7-vanilla:${TAG}
}

build_and_push nv-latest ${BRANCH_NAME}-nv-amd64-${SHORT_SHA} amd64
build_and_push nv-latest ${BRANCH_NAME}-nv-arm64-${SHORT_SHA} arm64

create_manifest_and_push(){
  TAG=$1
  VARIANT=$2

  docker manifest create \
    gcr.io/${PROJECT_ID}/yolov7-vanilla:${TAG} \
    gcr.io/${PROJECT_ID}/yolov7-vanilla:${BRANCH_NAME}-${VARIANT}amd64-${SHORT_SHA} \
    gcr.io/${PROJECT_ID}/yolov7-vanilla:${BRANCH_NAME}-${VARIANT}arm64-${SHORT_SHA}

  docker manifest annotate \
    gcr.io/${PROJECT_ID}/yolov7-vanilla:${TAG} \
    gcr.io/${PROJECT_ID}/yolov7-vanilla:${BRANCH_NAME}-${VARIANT}amd64-${SHORT_SHA} \
    --os=linux

  docker manifest annotate \
    gcr.io/${PROJECT_ID}/yolov7-vanilla:${TAG} \
    gcr.io/${PROJECT_ID}/yolov7-vanilla:${BRANCH_NAME}-${VARIANT}arm64-${SHORT_SHA} \
    --os=linux \
    --arch=arm64 \
    --variant=v8

  docker manifest push gcr.io/${PROJECT_ID}/yolov7-vanilla:${TAG}
}

create_manifest_and_push ${BRANCH_NAME}-nv-${SHORT_SHA} "nv-"

if [ ${BRANCH_NAME} == 'main' ]; then
  create_manifest_and_push nv-latest "nv-"
fi