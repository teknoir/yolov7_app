#!/usr/bin/env bash
set -eo pipefail
set -x

export BASE_IMAGE=${BASE_IMAGE:-"nvcr.io/nvidia/l4t-ml:r36.2.0-py3"}
export BRANCH_NAME=${BRANCH_NAME:-"main"}
export SHORT_SHA=${SHORT_SHA:-$(git rev-parse --short HEAD)}
export IMAGE_NAME=${IMAGE_NAME:-"us-docker.pkg.dev/teknoir/gcr.io/yolov7-base"}
export TAG=${TAG:-"l4t-r36.2.0"}

docker buildx build \
  --builder mybuilder \
  --platform=linux/arm64/v8 \
  --push \
  --label "git-commit=${SHORT_SHA}" \
  --build-arg=BASE_IMAGE=${BASE_IMAGE} \
  --tag ${IMAGE_NAME}:${TAG}-${BRANCH_NAME}-${SHORT_SHA} \
  --file l4t.Dockerfile \
  .

if [ ${BRANCH_NAME} == 'main' ]; then
  docker buildx build \
    --builder mybuilder \
    --platform=linux/arm64/v8 \
    --push \
    --label "git-commit=${SHORT_SHA}" \
    --build-arg=BASE_IMAGE=${BASE_IMAGE} \
    --tag ${IMAGE_NAME}:${TAG} \
    --file l4t.Dockerfile \
    .
fi