set -eo pipefail

export SHORT_SHA=${SHORT_SHA:-"head"}
export BRANCH_NAME=${BRANCH_NAME:-"local"}
export PROJECT_ID=${PROJECT_ID:-"teknoir"}

build_and_push()
{
  TAG=$1
  docker pull gcr.io/teknoir/yolov7_with_img_stack_vanilla:${TAG}
  docker buildx build \
    --no-cache \  
    --build-arg=BASE_IMAGE=gcr.io/teknoir/yolov7_with_img_stack_vanilla:${TAG} \
    --build-arg=MODEL_NAME=yolov7_with_img_stack \
    --build-arg=TRAINING_DATASET=cocoa \
    --build-arg=IMG_SIZE=416 \
    --build-arg=WEIGHTS_FILE=yolov7-tiny.pt \
    --build-arg=CLASS_NAMES_FILE=classes.names \
    --platform=linux/arm64 \
    --label "git-commit=${SHORT_SHA}" \
    --push \
    -t gcr.io/${PROJECT_ID}/yolov7:${TAG}-${BRANCH_NAME}-${SHORT_SHA} \
    -f ./vanilla.Dockerfile .
}

build_and_push l4tr34.1.1
build_and_push l4tr32.7.1
build_and_push l4tr32.7.2