#!/bin/bash

set -e

python3 -m tf2onnx.convert \
--saved-model $1 \
--opset $3 \
--output $2
