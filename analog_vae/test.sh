#!/bin/bash -x
#
# usage:
# ./test.sh model in.s16 out.s16

OPUS=${HOME}/opus
PATH=${PATH}:${OPUS}

features_in=features.f32
features_out=out.f32

if [ ! -f $1 ]; then
    echo "can't find $1"
    exit 1
fi
if [ ! -f $2 ]; then
    echo "can't find $2"
    exit 1
fi

lpcnet_demo -features $2 ${features_in}
python3 ./test_rdovae.py $1 ${features_in} ${features_out}
lpcnet_demo -fargan-synthesis ${features_out} $3


