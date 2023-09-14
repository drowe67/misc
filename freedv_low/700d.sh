#!/bin/bash -x
# 700d.sh
# Generate FreeDV 700D BER and PER curves for comparison

CODEC2=$HOME/codec2/build_linux/src
PATH=$PATH:$CODEC2

# length of simulation
MPP_SAMPLES=fast_fading_samples.float
TEST_SEC=300
RESULTS="700d_results.txt"
No_list='-18 -19 -20 -21 -22 -23 -24 -26'

# Generate fading data files
if [ ! -f $MPP_SAMPLES ]; then
    echo "Generating fading files ......"
    cmd='pkg load signal; ch_fading("'${MPP_SAMPLES}'", 8000, 1.0, 8000*'${TEST_SEC}')'
    octave --no-gui -qf --eval "$cmd"
    [ ! $? -eq 0 ] && { echo "octave failed to run correctly .... exiting"; exit 1; }
fi

# Run 700D
rm $RESULTS
results_ch=$(mktemp)
results=$(mktemp)
for No in $No_list
do
    ofdm_mod --in /dev/zero --ldpc 1 --testframes $TEST_SEC --txbpf | \
    ch - - --No ${No} -f -10 --mpp --fading_dir . 2> $results_ch | \
    ofdm_demod --out /dev/null --testframes --verbose 2 --ldpc 1 2> $results
    snrdB=$(cat $results_ch | grep "SNR3k(dB)" | tr -s ' ' | cut -d' ' -f3)
    cber=$(cat $results | grep "Coded BER" | tr -s ' ' | cut -d' ' -f3)
    cper=$(cat $results | grep "Coded PER" | tr -s ' ' | cut -d' ' -f3)
    echo $snrdB $cber $cper >> $RESULTS
done
