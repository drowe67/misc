#!/bin/bash

CODEC2_PATH=$HOME/codec2-dev
PATH=$PATH:$CODEC2_PATH/build_linux/src:$CODEC2_PATH/build_linux/misc

K=10
M=4096

train=$1
prefix=$2
vqtrain $1 $K $M ${prefix}_vq1.f32 --split -s 0.0001 -r res1.f32 1>${prefix}_var1.txt
vqtrain res1.f32 $K $M ${prefix}_vq2.f32 --split -s 0.0001 -r res2.f32 1>${prefix}_var2.txt
vqtrain res2.f32 $K $M ${prefix}_vq3.f32 --split -s 0.0001 1>${prefix}_var3.txt
