#!/bin/bash -x
# ratek_resampler.sh
# David Rowe Sep 2022
#
# Support for rate K resampler experiments see doc/ratek_resampler

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

function batch_process {
  fullfile=$1
  filename=$(basename -- "$fullfile")
  filename="${filename%.*}"
  batch_opt=$2
  outname=$3
  c2sim_opt=$4
  tmp=$(mktemp)
  
  echo "ratek3_batch;" \
       "ratek3_batch_tool(\"${filename}\", "\
                          "'A_out',\"${filename}_a.f32\"," \
                          "'H_out',\"${filename}_h.f32\"," \
                          "${batch_opt},'logfn',\"${tmp}\"); quit;" \
  | octave -p ${CODEC2_PATH}/octave -qf
  c2sim $fullfile --hpf --phase0 --postfilter --amread ${filename}_a.f32 --hmread ${filename}_h.f32 -o - \
  ${c2sim_opt} | sox -t .s16 -r 8000 -c 1 - ${out_dir}/${filename}_${outname}.wav
  printf "%-10s %-20s %4.2f\n" ${filename} ${outname} $(cat ${tmp}) >> ${out_dir}/zlog.txt
}

# 231028: Testing K=20 and K=80 VQs, no decimation in time (10ms frames)
function vq_test_231028() {
  fullfile=$1
  filename=$(basename -- "$fullfile")
  filename="${filename%.*}"
  extension="${filename##*.}"
  mkdir -p $out_dir
  if [ 1 -eq 0 ]; then
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
fi
  # Amps Nb filtered, phase0, rate K=20 resampling, normalise energy
  batch_process $fullfile "'norm_en','Nb',100'" "4_k80"  

  # K=80, 2 x 12 VQ
  batch_process $fullfile "'norm_en','Nb',100', \
  'vq1','train_k80_vq1.f32', \
  'vq2','train_k80_vq2.f32'"  "5_k80_vq"

  cat $fullfile | hpf | c2enc 3200 - - | c2dec 3200 - - | sox -t .s16 -r 8000 -c 1 - ${out_dir}/${filename}_6_3200.wav
  
}

# generate amp postfiltered rate K training material (20 element vectors) from source speech file, 
# norm enenergy and Nb=20 filtering
function gen_train_b() {
  fullfile=$1
  filename=$(basename -- "$fullfile")
  extension="${filename##*.}"
  filename="${filename%.*}"
  filename_b=$2
 
  c2sim $fullfile --hpf --modelout ${filename}_model.bin
  echo "ratek3_batch; ratek3_batch_tool(\"${filename}\",'B_out',\"${filename_b}\", \
        'K',20,'norm_en'); quit;" \
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

  c2sim $fullfile --hpf --modelout ${filename}_model.bin
  echo "ratek3_batch; ratek3_batch_tool(\"${filename}\",'Y_out',\"${filename_y}\", \
        'norm_en','Nb',100); quit;" \
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
    gen_train_y)
        gen_train_y $2 $3
        ;;
    gen_train_comp)
        gen_train_comp $2 $3
        ;;
    vq_test_231028)
        vq_test_231028 ${CODEC2_PATH}/raw/big_dog.raw
        vq_test_231028 ${CODEC2_PATH}/raw/two_lines.raw
        #comp_test_230323 ../raw/forig.raw     
        #comp_test_230323 ../raw/pickle.raw
        #comp_test_230323 ../raw/tea.raw
        #comp_test_230323 ../raw/kristoff.raw        
        #comp_test_230323 ../raw/ve9qrp_10s.raw     
        #comp_test_230323 ../raw/mmt1.raw     
        ;;
    esac
else
  echo "usage:
  echo "  ./ratek_resampler.sh command [options ...]""
fi

