#!/bin/bash -x
# analog.sh
# David Rowe Jan 2024
#
# Shell script that ties analog project prcoessing together

CODEC2_PATH=$HOME/codec2-dev
PATH=$PATH:$CODEC2_PATH/build_linux/src:$CODEC2_PATH/build_linux/misc

# bunch of options we can set via variables
out_dir="${out_dir:-ratek_out}"
extract_options="${extract_options:-}"
options="${options:-}"
Nb=20

which c2sim || { printf "\n**** Can't find c2sim - check CODEC2_PATH **** \n\n"; exit 1; }

function batch_process_ml2 {
  fullfile=$1
  filename=$(basename -- "$fullfile")
  filename="${filename%.*}"
  batch_opt=$2
  outname=$3
  c2sim_opt=$4
  tmp=$(mktemp)

  echo "batch_process_ml2 ------------------------------"

  # if something bombs make sure we rm previous sample to indicate problem
  rm -f ${out_dir}/${filename}_${outname}.wav

  echo "linear_batch;" \
       "linear_batch_ml_out(\"${filename}\", "\
                          "'A_out',\"${filename}_a.f32\"," \
                          "'H_out',\"${filename}_h.f32\"," \
                          "${batch_opt}); quit;" \
  | octave-cli -qf
  c2sim $fullfile --hpf --phase0 --postfilter --amread ${filename}_a.f32 --hmread ${filename}_h.f32 -o - \
  ${c2sim_opt} | sox -t .s16 -r 8000 -c 1 - ${out_dir}/${filename}_${outname}.wav
}

# Baseline autoencoder testing, refactored version of misc/manifold
function test_240104 {
  fullfile=$1
  filename=$(basename -- "$fullfile")
  filename="${filename%.*}"
  extension="${filename##*.}"
  mkdir -p $out_dir

  c2sim $fullfile --hpf --modelout ${filename}_model.bin --dump ${filename}

  # orig amp and phase
  c2sim $fullfile --hpf --modelout ${filename}_model.bin -o - | \
  sox -t .s16 -r 8000 -c 1 - ${out_dir}/${filename}_1_out.wav
 
  # Use ML inference to recover y_hat from b
  # note y is used for energy side information and measuring SD, shape of y_hat is inferred from b. 
  echo "linear_batch;" \
       "linear_batch_ml_in(\"${filename}\", 'Nb',100, 'Y_out', \"${filename}_y.f32\"); quit;" | octave-cli -qf
  echo "linear_batch;" \
       "linear_batch_ml_in(\"${filename}\", 'B_out', \"${filename}_b.f32\"); quit;" | octave-cli -qf
  python3 ../manifold/manifold.py ${filename}_b.f32 ${filename}_y.f32 --inference ../manifold/model1.pt --noplot --out_file ${filename}_y_hat.f32
  batch_process_ml2 $fullfile "'Y_in','${filename}_y.f32','Y_hat_in','${filename}_y_hat.f32'" "3_k80_y_hat"

  # Codec 2 3200 anchor
  cat $fullfile | hpf | c2enc 3200 - - | c2dec 3200 - - | sox -t .s16 -r 8000 -c 1 - ${out_dir}/${filename}_8_3200.wav 
}

if [ $# -gt 0 ]; then
  case $1 in
     test_240104)
        test_240104 ${CODEC2_PATH}/raw/big_dog.raw
        #ml_test_231126 ${CODEC2_PATH}/raw/two_lines.raw
        #ml_test_231126 ${CODEC2_PATH}/raw/hts1a.raw
        #ml_test_231126 ${CODEC2_PATH}/raw/kristoff.raw
        #ml_test_231126 ${CODEC2_PATH}/raw/mmt1.raw
      ;;
    esac
else
  echo "usage:
  echo "  ./analog.sh command [options ...]""
fi

