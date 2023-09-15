#!/bin/bash -x
# ofdm_c.sh
# Generate BER and PER curves for comparison from existing OFDM C implementations

CODEC2=$HOME/codec2/build_linux/src
PATH=$PATH:$CODEC2

# length of simulation
MPP_SAMPLES=fast_fading_samples.float
TEST_SEC=300

# list of test noise levels (ch tool parameter)
No_awgn='-12 -12.5 -13 -13.5 -14 -15 -16 -18 -19 -20'
No_mpp='-18 -19 -20 -21 -22 -23 -24 -26'

function run_sim {
    results=$1
    mode=$2
    channel=$3
    No_list=$4
    rm $results
    log_ch=$(mktemp)
    log_demod=$(mktemp)

    for No in $No_list
    do
        ofdm_mod --in /dev/zero --mode $mode --ldpc 1 --testframes $TEST_SEC --txbpf | \
        ch - - --No ${No} $channel --fading_dir . 2> $log_ch | \
        ofdm_demod --out /dev/null --mode $mode --testframes --verbose 2 --ldpc 1 2> $log_demod
        snrdB=$(cat $log_ch | grep "SNR3k(dB)" | tr -s ' ' | cut -d' ' -f3)
        uber=$(cat $log_demod | grep "BER\." | tr -s ' ' | cut -d' ' -f2)
        cber=$(cat $log_demod | grep "Coded BER" | tr -s ' ' | cut -d' ' -f3)
        cper=$(cat $log_demod | grep "Coded PER" | tr -s ' ' | cut -d' ' -f3)
        echo $snrdB $uber $cber $cper >> $results
    done
}

# Generate fading data files
if [ ! -f $MPP_SAMPLES ]; then
    echo "Generating fading files ......"
    cmd='pkg load signal; ch_fading("'${MPP_SAMPLES}'", 8000, 1.0, 8000*'${TEST_SEC}')'
    octave --no-gui -qf --eval "$cmd"
    [ ! $? -eq 0 ] && { echo "octave failed to run correctly .... exiting"; exit 1; }
fi

run_sim "700d_awgn.txt" "700D" ""      "${No_awgn}"
run_sim "700d_mpp.txt"  "700D" "--mpp" "${No_mpp}"
run_sim "700e_awgn.txt" "700E" ""      "${No_awgn}"
run_sim "700e_mpp.txt"  "700E" "--mpp" "${No_mpp}"
