set -eo pipefail

export SHORT_SHA=${SHORT_SHA:-"head"}
export BRANCH_NAME=${BRANCH_NAME:-"local"}
export PROJECT_ID=${PROJECT_ID:-"teknoir"}

build_and_push()
{
  BASE_TAG=$1
  TAG=$2
  ARCH=$3
  docker pull --platform=linux/${ARCH} gcr.io/teknoir/yolov7_with_img_stack:${BASE_TAG}

  docker buildx build \
    --build-arg=BASE_IMAGE=gcr.io/teknoir/yolov7_with_img_stack:${BASE_TAG} \
    --build-arg=MODEL_NAME=yolov7_stacking_vanilla \
    --build-arg=TRAINING_DATASET=cocoa \
    --build-arg=IMG_SIZE=640 \
    --build-arg=WEIGHTS_FILE=yolov7-tiny.pt \
    --build-arg=CLASS_NAMES_FILE=classes.names \
    --platform=linux/${ARCH} \
    --label "git-commit=${SHORT_SHA}" \
    --push \
    -t gcr.io/${PROJECT_ID}/yolov7_stacking_vanilla:${TAG} \
    -f ./vanilla.Dockerfile .
}

build_and_push latest ${BRANCH_NAME}-amd64-${SHORT_SHA} amd64
build_and_push latest ${BRANCH_NAME}-arm64-${SHORT_SHA} arm64

create_manifest_and_push(){
  TAG=$1
  VARIANT=$2

  docker manifest create \
    gcr.io/${PROJECT_ID}/yolov7_stacking_vanilla:${TAG} \
    gcr.io/${PROJECT_ID}/yolov7_stacking_vanilla:${BRANCH_NAME}-${VARIANT}amd64-${SHORT_SHA} \
    gcr.io/${PROJECT_ID}/yolov7_stacking_vanilla:${BRANCH_NAME}-${VARIANT}arm64-${SHORT_SHA}

  docker manifest annotate \
    gcr.io/${PROJECT_ID}/yolov7_stacking_vanilla:${TAG} \
    gcr.io/${PROJECT_ID}/yolov7_stacking_vanilla:${BRANCH_NAME}-${VARIANT}amd64-${SHORT_SHA} \
    --os=linux

  docker manifest annotate \
    gcr.io/${PROJECT_ID}/yolov7_stacking_vanilla:${TAG} \
    gcr.io/${PROJECT_ID}/yolov7_stacking_vanilla:${BRANCH_NAME}-${VARIANT}arm64-${SHORT_SHA} \
    --os=linux \
    --arch=arm64 \
    --variant=v8

  docker manifest push gcr.io/${PROJECT_ID}/yolov7_stacking_vanilla:${TAG}
}

create_manifest_and_push ${BRANCH_NAME}-${SHORT_SHA} ""

if [ ${BRANCH_NAME} == 'main' ]; then
  create_manifest_and_push latest ""
fi
