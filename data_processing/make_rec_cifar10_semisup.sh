#!/bin/bash

# usage:  ./make_rec_cifar10_semisup.sh "/tanData/datasets/cifar10" "train_valid_sup" "cifar10_train_valid_sup"

DATA_DIR=$1
data_name=$2
list_name=$3
MX_DIR=/mxnet

# clean stuffs
rm -rf ${DATA_DIR}/${list_name}.*

# make list for all classes
python ${MX_DIR}/tools/im2rec.py --list True --exts '.png' --recursive True ${DATA_DIR}/${list_name} ${DATA_DIR}/${data_name}

# make .rec file for all classes
python ${MX_DIR}/tools/im2rec.py --exts '.png' --quality 95 --num-thread 16 --color 1 ${DATA_DIR}/${list_name} ${DATA_DIR}/${data_name}