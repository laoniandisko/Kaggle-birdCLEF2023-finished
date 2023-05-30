#!/bin/bash

GPU=$1
VAL_OUT_DIR=$2
CONFIG=$3
FOLD=$4
DATA_DIR=$5
PREFIX=$6
RESUME=$7

PYTHONPATH=.  python -u train_classifier.py --gpu $GPU  \
 --config configs/${CONFIG}.json  --workers 8 --test_every 1 \
 --val-dir $VAL_OUT_DIR  --prefix $PREFIX --fold $FOLD --freeze-epochs 0 --data-dir $DATA_DIR --resume weights/$RESUME