#!/bin/bash -x
#
# Process an audio file with test_rdovae.py

OPUS=${HOME}/opus
PATH=${PATH}:${OPUS}

features_in=features.f32
features_out=out.f32

if [ $# -lt 3 ]; then
    echo "usage (write output to file):"
    echo "  ./test.sh model in.s16 out.s16 [optional test_rdovae.py args]"
    echo "usage (play output with aplay):"
    echo "  ./test.sh model in.s16 - [optional test_rdovae.py args]"
    exit 1
fi
if [ ! -f $1 ]; then
    echo "can't find $1"
    exit 1
fi
if [ ! -f $2 ]; then
    echo "can't find $2"
    exit 1
fi

model=$1
input_speech=$2
output_speech=$3
features_in=$(mktemp)
features_out=$(mktemp)

# eat first 3 args before passing rest to test_rdovae.py in $@
shift; shift; shift

lpcnet_demo -features ${input_speech} ${features_in}
python3 ./test_rdovae.py ${model} ${features_in} ${features_out} "$@"
if [ ! $output_speech == "-" ]; then
    lpcnet_demo -fargan-synthesis ${features_out} ${output_speech}
else
    tmp=$(mktemp)
    lpcnet_demo -fargan-synthesis ${features_out} $tmp
    aplay $tmp -r 16000 -f S16_LE
fi
