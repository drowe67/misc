#!/bin/bash -x
# ratek_resampler.sh
# David Rowe Oct 2023
#
# Support for manifold experiments see misc/manifold

CODEC2_PATH=$HOME/codec2-dev
PATH=$PATH:$CODEC2_PATH/build_linux/src:$CODEC2_PATH/build_linux/misc

# bunch of options we can set via variables
K="${K:-30}"
M="${M:-4096}"
Kst="${Kst:-0}"  # first index 
Ksp="${Ksp:-9}"  # last element of first vector in split
Ken="${Ken:-29}" # last index max K-1
out_dir="${out_dir:-ratek_out}"
extract_options="${extract_options:-}"
options="${options:-}"
mbest="${mbest:-no}"
removemean="${removemean:---removemean}"
lower=${lower:-10}
meanl2=${meanl2:-}
dr=${dr:-100}
drlate=${drlate:-}
stage2="${stage2:-yes}"
stage3="${stage3:-no}"
Nb=20

which c2sim || { printf "\n**** Can't find c2sim - check CODEC2_PATH **** \n\n"; exit 1; }
which vqtrain || { printf "\n**** Can't find vqtrain, build codec2-dev with cmake -DUNITTEST=1 .. **** \n\n"; exit 1; }

function batch_process {
  fullfile=$1
  filename=$(basename -- "$fullfile")
  filename="${filename%.*}"
  batch_opt=$2
  outname=$3
  c2sim_opt=$4
  tmp=$(mktemp)

  # if something bombs make sure we rm previous sample to indicate problem
  rm -f ${out_dir}/${filename}_${outname}.wav

  echo "ratek3_batch;" \
       "ratek3_batch_tool(\"${filename}\", "\
                          "'A_out',\"${filename}_a.f32\"," \
                          "'H_out',\"${filename}_h.f32\"," \
                          "${batch_opt},'logfn',\"${tmp}\"); quit;" \
  | octave-cli -qf
  c2sim $fullfile --hpf --phase0 --postfilter --amread ${filename}_a.f32 --hmread ${filename}_h.f32 -o - \
  ${c2sim_opt} | sox -t .s16 -r 8000 -c 1 - ${out_dir}/${filename}_${outname}.wav
  printf "%-10s %-20s %4.2f\n" ${filename} ${outname} $(cat ${tmp}) >> ${out_dir}/zlog.txt
}

function batch_process_ml {
  fullfile=$1
  filename=$(basename -- "$fullfile")
  filename="${filename%.*}"
  batch_opt=$2
  outname=$3
  c2sim_opt=$4
  tmp=$(mktemp)

  echo "batch_process_ml ------------------------------"

  # if something bombs make sure we rm previous sample to indicate problem
  rm -f ${out_dir}/${filename}_${outname}.wav

  echo "ratek3_batch;" \
       "ratek80_batch_tool(\"${filename}\", "\
                          "'A_out',\"${filename}_a.f32\"," \
                          "'H_out',\"${filename}_h.f32\"," \
                          "${batch_opt}); quit;" \
  | octave-cli -qf
  c2sim $fullfile --hpf --phase0 --postfilter --amread ${filename}_a.f32 --hmread ${filename}_h.f32 -o - \
  ${c2sim_opt} | sox -t .s16 -r 8000 -c 1 - ${out_dir}/${filename}_${outname}.wav
}

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

  echo "ratek3_batch;" \
       "ratek80_batch_ml_out(\"${filename}\", "\
                          "'A_out',\"${filename}_a.f32\"," \
                          "'H_out',\"${filename}_h.f32\"," \
                          "${batch_opt}); quit;" \
  | octave-cli -qf
  c2sim $fullfile --hpf --phase0 --postfilter --amread ${filename}_a.f32 --hmread ${filename}_h.f32 -o - \
  ${c2sim_opt} | sox -t .s16 -r 8000 -c 1 - ${out_dir}/${filename}_${outname}.wav
}

# 231126: Testing K=80 ML experiment from manifold.py
function ml_test_231126 {
  fullfile=$1
  filename=$(basename -- "$fullfile")
  filename="${filename%.*}"
  extension="${filename##*.}"
  mkdir -p $out_dir
#: <<'END'
  c2sim $fullfile --hpf --modelout ${filename}_model.bin --dump ${filename}

  # orig amp and phase
  c2sim $fullfile --hpf --modelout ${filename}_model.bin -o - | \
  sox -t .s16 -r 8000 -c 1 - ${out_dir}/${filename}_1_out.wav
 
  # Nb=20 filtered (smoothed), rate K=20 resampling, input b for inference test
  batch_process_ml $fullfile "'norm_en','Nb',20,'prede','B_out','${filename}_b.f32'" "2_k20"  

  # No filtering, rate K=80 resampling to get y out for ideal y test 4_k80_y below
  batch_process_ml $fullfile "'norm_en','Nb',100,'prede','Y_out','${filename}_y.f32'" "3_k80"  

  # Test with ideal y, should be identical to 3_k80 
  batch_process_ml $fullfile "'norm_en','Nb',100,'prede','Y_in','${filename}_y.f32'" "4_k80_y"  

  # Use ML inference to recover y_hat from b, note ${filename}_y.f32 is not used (non-optional cmd line arg)
  python3 manifold.py ${filename}_b.f32 ${filename}_y.f32 --inference model1.pt --noplot --out_file ${filename}_y_hat.f32
  batch_process_ml $fullfile "'norm_en','Nb',100,'prede','Y_in','${filename}_y_hat.f32'" "5_k80_y_hat"
#END
  # Use seprate batch process functions for enc and dec to double check results, should be identical to 5
  echo "ratek3_batch;" \
       "ratek80_batch_ml_in(\"${filename}\", 'B_out', \"${filename}_b.f32\"); quit;" \
  | octave-cli -qf
  python3 manifold.py ${filename}_b.f32 ${filename}_y.f32 --inference model1.pt --noplot --out_file ${filename}_y_hat.f32
  batch_process_ml2 $fullfile "'Y_in','${filename}_y_hat.f32'" "6_k80_y_hat"

  cat $fullfile | hpf | c2enc 3200 - - | c2dec 3200 - - | sox -t .s16 -r 8000 -c 1 - ${out_dir}/${filename}_8_3200.wav 
}


# 231031: Testing a few different K, no decimation in time (10ms frames)
function vq_test_231031() {
  fullfile=$1
  filename=$(basename -- "$fullfile")
  filename="${filename%.*}"
  extension="${filename##*.}"
  mkdir -p $out_dir
  
  c2sim $fullfile --hpf --modelout ${filename}_model.bin

  # orig amp and phase
  c2sim $fullfile --hpf --modelout ${filename}_model.bin -o - | \
  sox -t .s16 -r 8000 -c 1 - ${out_dir}/${filename}_1_out.wav
 
  # Amps Nb=20 filtered, phase0, rate K=20 resampling, normalise energy
  batch_process $fullfile "'K',20,'norm_en'" "2_k20"  

  # Amps Nb=40 filtered, phase0, rate K=40 resampling, normalise energy
  batch_process $fullfile "'K',40,'norm_en','Nb',40" "3_k40"  

  # No filtering, phase0, rate K=80 resampling, normalise energy
  batch_process $fullfile "'norm_en','Nb',100, " "4_k80"  

  cat $fullfile | hpf | c2enc 3200 - - | c2dec 3200 - - | sox -t .s16 -r 8000 -c 1 - ${out_dir}/${filename}_7_3200.wav 
}

# 231028: Testing K=20 and K=80 VQs, no decimation in time (10ms frames)
function vq_test_231028() {
  fullfile=$1
  filename=$(basename -- "$fullfile")
  filename="${filename%.*}"
  extension="${filename##*.}"
  mkdir -p $out_dir
  
  c2sim $fullfile --hpf --modelout ${filename}_model.bin

  # orig amp and phase
  c2sim $fullfile --hpf --modelout ${filename}_model.bin -o - | \
  sox -t .s16 -r 8000 -c 1 - ${out_dir}/${filename}_1_out.wav
 
  # Amps Nb=20 filtered, phase0, rate K=20 resampling, normalise energy
  batch_process $fullfile "'K',20,'norm_en'" "2_k20"  

  # K=20, 2 x 12 VQ
  batch_process $fullfile "'K',20,'norm_en', \
  'vq1','train_k20_vq1.f32', \
  'vq2','train_k20_vq2.f32'"  "3_k20_vq"

  # K=40, Nb=40
  batch_process $fullfile "'K',40,'norm_en','Nb',40" "4_k40"

  # K=40, 2 x 12 VQ
  batch_process $fullfile "'K',40,'norm_en','Nb',40,'verbose', \
  'vq1','train_k40_vq1.f32', \
  'vq2','train_k40_vq2.f32'"  "5_k40_vq"

  # No filtering, phase0, rate K=80 resampling, normalise energy
  batch_process $fullfile "'norm_en','Nb',100" "6_k80"  

  # K=80, 2 x 12 VQ
  batch_process $fullfile "'norm_en','Nb',100, 'verbose', \
  'vq1','train_k80_vq1.f32', \
  'vq2','train_k80_vq2.f32'"  "7_k80_vq"

  cat $fullfile | hpf | c2enc 3200 - - | c2dec 3200 - - | sox -t .s16 -r 8000 -c 1 - ${out_dir}/${filename}_8_3200.wav 
}

# generate rate K=20, Nb=20 training data
function gen_train_b() {
  fullfile=$1
  filename=$(basename -- "$fullfile")
  extension="${filename##*.}"
  filename="${filename%.*}"
  filename_b=$2
 
  c2sim $fullfile --hpf --modelout ${filename}_model.bin ${options}
  echo "ratek3_batch; ratek3_batch_tool(\"${filename}\",'B_out',\"${filename_b}\", \
        'K',20,'norm_en'); quit;" \
  | octave -p ${CODEC2_PATH}/octave -qf
}

# generate rate K=40, Nb=40 training data
function gen_train_b40() {
  fullfile=$1
  filename=$(basename -- "$fullfile")
  extension="${filename##*.}"
  filename="${filename%.*}"
  filename_b=$2
 
  c2sim $fullfile --hpf --modelout ${filename}_model.bin
  echo "ratek3_batch; ratek3_batch_tool(\"${filename}\",'B_out',\"${filename_b}\", \
        'K',40,'norm_en','Nb',40); quit;" \
  | octave -p ${CODEC2_PATH}/octave -qf
}

# Generate oversampled K=80 (79 element vectors) training material from source speech file
# Energy is normalised and no filtering.
function gen_train_y() {
  fullfile=$1
  filename=$(basename -- "$fullfile")
  extension="${filename##*.}"
  filename="${filename%.*}"
  filename_y=$2

  c2sim $fullfile --hpf --modelout ${filename}_model.bin ${options}
  echo "ratek3_batch; ratek3_batch_tool(\"${filename}\",'Y_out',\"${filename_y}\", \
        'norm_en','Nb',100); quit;" \
  | octave -p ${CODEC2_PATH}/octave -qf
}
      
# generate rate K=20, Nb=20 training data for ML experiment
function gen_train_b_ml() {
  fullfile=$1
  filename=$(basename -- "$fullfile")
  extension="${filename##*.}"
  filename="${filename%.*}"
  filename_b=$2
 
  c2sim $fullfile --hpf --prede --modelout ${filename}_model.bin ${options}
  echo "ratek3_batch; ratek3_batch_tool(\"${filename}\",'B_out',\"${filename_b}\", \
        'K',20,'norm_en','append_Wo_v'); quit;" \
  | octave -p ${CODEC2_PATH}/octave -qf
}

# Generate oversampled K=80 (79 element vectors) training material for ML experiment
function gen_train_y_ml() {
  fullfile=$1
  filename=$(basename -- "$fullfile")
  extension="${filename##*.}"
  filename="${filename%.*}"
  filename_y=$2

  c2sim $fullfile --hpf --prede --modelout ${filename}_model.bin ${options}
  echo "ratek3_batch; ratek3_batch_tool(\"${filename}\",'Y_out',\"${filename_y}\", \
        'Nb',100,'norm_en'); quit;" \
  | octave -p ${CODEC2_PATH}/octave -qf
}

function log2 {
    local x=0
    for (( y=$1-1 ; $y > 0; y >>= 1 )) ; do
        let x=$x+1
    done
    echo $x
}

# Train using LBG, produces nice curves of VQs SD versus bits 
function train_lbg() {
  fullfile=$1
  filename=$(basename -- "$fullfile")
  extension="${filename##*.}"
  filename="${filename%.*}"

  filename_out=${filename}_lbg
  if [ $# -eq 2 ]; then
    filename_out=$2
  fi
  
  # remove mean, extract columns from training data
  extract -t $K -s $Kst -e $Ken --lower $lower $removemean $meanl2 \
  --dynamicrange $dr $extract_options $drlate --writeall $fullfile ${filename_out}_nomean.f32

  # train 2 stages - LBG
  vqtrain ${filename_out}_nomean.f32 $K $M --st $Kst --en $Ken -s 1e-3 ${filename_out}_vq1.f32 -r res1.f32 --split > ${filename_out}_res1.txt
  if [ "$stage2" == "yes" ]; then
    vqtrain res1.f32 $K $M --st $Kst --en $Ken -s 1e-3 ${filename_out}_vq2.f32 -r res2.f32 --split > ${filename_out}_res2.txt
  fi
  if [ "$stage3" == "yes" ]; then
    vqtrain res2.f32 $K $M --st $Kst --en $Ken -s 1e-3 ${filename_out}_vq3.f32 --split > ${filename_out}_res3.txt
  fi
      
  # optionally compare stage3 search with mbest
  if [ "$mbest" == "yes" ]; then
    tmp=$(mktemp)
    results=${filename_out}_mbest3.txt
    rm ${results}
    log2M=$(log2 $M)
    for alog2M in $(seq 1 $log2M)
    do
      aM=$(( 2 ** $alog2M ))
      vqtrain res2.f32 $K $aM --st $Kst --en $Ken -s 1e-3 ${filename_out}_vq3.f32 --split > /dev/null
      cat ${filename_out}_nomean.f32 | \
          vq_mbest --mbest 5 -k $K -q ${filename_out}_vq1.f32,${filename_out}_vq2.f32,${filename_out}_vq3.f32 2>${tmp} >> /dev/null
      echo -n "$aM " >> ${results}
      cat ${tmp} | grep var | cut -d' ' -f 2 >> ${results}
    done
  fi

}

if [ $# -gt 0 ]; then
  case $1 in
    gen_train_b)
        gen_train_b $2 $3
        ;;
    gen_train_b40)
        gen_train_b40 $2 $3
        ;;
    gen_train_y)
        gen_train_y $2 $3
        ;;
    gen_train_b_ml)
        gen_train_b_ml $2 $3
        ;;
    gen_train_y_ml)
        gen_train_y_ml $2 $3
        ;;
     train_lbg)
        train_lbg $2 $3
        ;;
    vq_test_231028)
        vq_test_231028 ${CODEC2_PATH}/raw/big_dog.raw
        vq_test_231028 ${CODEC2_PATH}/raw/two_lines.raw
        ;;
    vq_test_231031)
        vq_test_231031 ${CODEC2_PATH}/raw/big_dog.raw
        vq_test_231031 ${CODEC2_PATH}/raw/two_lines.raw
        ;;   
    ml_test_231126)
        ml_test_231126 ${CODEC2_PATH}/raw/big_dog.raw
        #ml_test_231126 ${CODEC2_PATH}/raw/two_lines.raw
        #ml_test_231126 ${CODEC2_PATH}/raw/hts1a.raw
        #ml_test_231126 ${CODEC2_PATH}/raw/kristoff.raw
        #ml_test_231126 ${CODEC2_PATH}/raw/mmt1.raw
      ;;
    esac
else
  echo "usage:
  echo "  ./ratek_resampler.sh command [options ...]""
fi

