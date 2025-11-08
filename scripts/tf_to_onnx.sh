#!/bin/bash

set -e

SAVED_MODEL_DIR="./models" || $1
OUTPUT_ONNX_MODEL="fpnlite320x320.onnx" || $2
OPSET=13 || $3

python3 -m tf2onnx.convert \
--saved-model $SAVED_MODEL_DIR \
--opset $OPSET \
--output $OUTPUT_ONNX_MODEL
