#!/bin/bash -x
# analog.sh
# David Rowe Jan 2024
#
# Shell script that ties analog project prcoessing together

CODEC2_PATH=$HOME/codec2-dev
PATH=$PATH:$CODEC2_PATH/build_linux/src:$CODEC2_PATH/build_linux/misc

# bunch of options we can set via variables
out_dir="${out_dir:-wav}"
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

# Autoencoder 4, dim 10 bottleneck, with VQ
function test_240118 {
  fullfile=$1
  filename=$(basename -- "$fullfile")
  filename="${filename%.*}"
  extension="${filename##*.}"
  mkdir -p $out_dir

#: <<'END'

  c2sim $fullfile --hpf --modelout ${filename}_model.bin --dump ${filename}

  # 1. orig amp and phase
  c2sim $fullfile --hpf --modelout ${filename}_model.bin -o - | \
  sox -t .s16 -r 8000 -c 1 - ${out_dir}/${filename}_1_out.wav
 
  # 3. Use ML inference to recover y_hat from b
  # note y is used for energy side information and measuring SD, shape of y_hat is inferred from b. 
  echo "linear_batch;" \
       "linear_batch_ml_in(\"${filename}\", 'Nb',100, 'Y_out', \"${filename}_y.f32\"); quit;" | octave-cli -qf
  echo "linear_batch;" \
       "linear_batch_ml_in(\"${filename}\", 'B_out', \"${filename}_b.f32\"); quit;" | octave-cli -qf
  python3 ../manifold/manifold.py ${filename}_b.f32 ${filename}_y.f32 --inference ../manifold/model1.pt --noplot --out_file ${filename}_y_hat.f32
  batch_process_ml2 $fullfile "'Y_in','${filename}_y.f32','Y_hat_in','${filename}_y_hat.f32'" "3_k80_y_hat"
  
#END

  # 5. Use prototype autoencoder 4 b_hat from b, then synthesise using manifold network
  python3 autoencoder1.py ${filename}_b.f32 --bottle_dim 10 --ncat 1 --nn 4 --norm --inference nn4_cat1.pt --out_file ${filename}_b_hat_nn4.f32 --noplot
  python3 ../manifold/manifold.py ${filename}_b_hat_nn4.f32 ${filename}_y.f32 --inference ../manifold/model1.pt --noplot --out_file ${filename}_y_hat_nn4.f32
  batch_process_ml2 $fullfile "'Y_in','${filename}_y.f32','Y_hat_in','${filename}_y_hat_nn4.f32'" "5_k80_b_hat_nn4"

  # 6. Use prototype autoencoder 4 with 24 bit VQ system to produce b_hat from b, then synthesise using manifold network
  python3 autoencoder1.py ${filename}_b.f32 --bottle_dim 10 --ncat 1 --nn 4 --norm --inference nn4_cat1.pt --write_latent ${filename}_l.f32 --noplot
  cat ${filename}_l.f32 | ~/codec2-dev/build_linux/misc/vq_mbest -k 10 -q vq1.f32,vq2.f32 --mbest 5 > ${filename}_l_hat.f32
  python3 autoencoder1.py ${filename}_b.f32 --bottle_dim 10 --ncat 1 --nn 4 --norm --inference nn4_cat1.pt --read_latent ${filename}_l_hat.f32 --out_file ${filename}_b_hat_vq24.f32 --noplot
  python3 ../manifold/manifold.py ${filename}_b_hat_vq24.f32 ${filename}_y.f32 --inference ../manifold/model1.pt --noplot --out_file ${filename}_y_hat_vq24.f32
  batch_process_ml2 $fullfile "'Y_in','${filename}_y.f32','Y_hat_in','${filename}_y_hat_vq24.f32'" "6_k80_b_hat_vq24"

  # As per 6 but 12 bit VQ
  cat ${filename}_l.f32 | ~/codec2-dev/build_linux/misc/vq_mbest -k 10 -q vq1.f32 --mbest 1 > ${filename}_l_hat.f32
  python3 autoencoder1.py ${filename}_b.f32 --bottle_dim 10 --ncat 1 --nn 4 --norm --inference nn4_cat1.pt --read_latent ${filename}_l_hat.f32 --out_file ${filename}_b_hat_vq12.f32 --noplot
  python3 ../manifold/manifold.py ${filename}_b_hat_vq12.f32 ${filename}_y.f32 --inference ../manifold/model1.pt --noplot --out_file ${filename}_y_hat_vq12.f32
  batch_process_ml2 $fullfile "'Y_in','${filename}_y.f32','Y_hat_in','${filename}_y_hat_vq12.f32'" "7_k80_b_hat_vq12"

  # Codec 2 3200 anchor
  cat $fullfile | hpf | c2enc 3200 - - | c2dec 3200 - - | sox -t .s16 -r 8000 -c 1 - ${out_dir}/${filename}_8_3200.wav 
}

# Baseline autoencoder testing, refactored version of misc/manifold
function test_240104 {
  fullfile=$1
  filename=$(basename -- "$fullfile")
  filename="${filename%.*}"
  extension="${filename##*.}"
  mkdir -p $out_dir

  c2sim $fullfile --hpf --modelout ${filename}_model.bin --dump ${filename}

  # 1. orig amp and phase
  c2sim $fullfile --hpf --modelout ${filename}_model.bin -o - | \
  sox -t .s16 -r 8000 -c 1 - ${out_dir}/${filename}_1_out.wav
 
  # 3. Use ML inference to recover y_hat from b
  # note y is used for energy side information and measuring SD, shape of y_hat is inferred from b. 
  echo "linear_batch;" \
       "linear_batch_ml_in(\"${filename}\", 'Nb',100, 'Y_out', \"${filename}_y.f32\"); quit;" | octave-cli -qf
  echo "linear_batch;" \
       "linear_batch_ml_in(\"${filename}\", 'B_out', \"${filename}_b.f32\"); quit;" | octave-cli -qf
  python3 ../manifold/manifold.py ${filename}_b.f32 ${filename}_y.f32 --inference ../manifold/model1.pt --noplot --out_file ${filename}_y_hat.f32
  batch_process_ml2 $fullfile "'Y_in','${filename}_y.f32','Y_hat_in','${filename}_y_hat.f32'" "3_k80_y_hat"
 
  # 4. Use prototype autoencoder 2 to produce b_hat from b, then synthesise using manifold network
  python3 autoencoder1.py ${filename}_b.f32 --bottle_dim 30 --ncat 4 --inference nn2_cat4.pt --nn 2 --noplot --out_file ${filename}_b_hat.f32
  python3 ../manifold/manifold.py ${filename}_b_hat.f32 ${filename}_y.f32 --inference ../manifold/model1.pt --noplot --out_file ${filename}_y_hat.f32
  batch_process_ml2 $fullfile "'Y_in','${filename}_y.f32','Y_hat_in','${filename}_y_hat.f32'" "4_k80_b_hat_nn2"
 
  # Codec 2 3200 anchor
  cat $fullfile | hpf | c2enc 3200 - - | c2dec 3200 - - | sox -t .s16 -r 8000 -c 1 - ${out_dir}/${filename}_8_3200.wav 
}

if [ $# -gt 0 ]; then
  case $1 in
    test_240104)
        test_240104 ${CODEC2_PATH}/raw/big_dog.raw
        test_240104 ${CODEC2_PATH}/raw/two_lines.raw
      ;;
    test_240118)
        test_240118 ${CODEC2_PATH}/raw/big_dog.raw
        test_240118 ${CODEC2_PATH}/raw/two_lines.raw
        test_240118 ${CODEC2_PATH}/raw/hts1a.raw
        test_240118 ${CODEC2_PATH}/raw/kristoff.raw
        test_240118 ${CODEC2_PATH}/raw/mmt1.raw
      ;;
    esac
else
  echo "usage:
  echo "  ./analog.sh command [options ...]""
fi

