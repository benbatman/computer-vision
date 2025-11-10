#!/bin/bash

set -e

python3 ./python/tf_text_graph_ssd.py \
--input $1 \ // .pb model file
--config $2 \ // config file
--output $3 // output file
