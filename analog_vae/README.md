# Setup

## LPCNet

```
git clone git@github.com:xiph/opus.git
cd opus
git checkout opus-ng
./autogen.sh
./configure --enable-dred
make
cd dnn
```
Initial test:
```
./lpcnet_demo -features input.pcm features.f32
./lpcnet_demo -fargan-synthesis features.f32 output.pcm
```
Playing on a remote machine:
```
scp deep.lan:opus/output.s16 /dev/stdout | aplay -f S16_LE -r 1600
```

## Training
```
python3 ./train_rdovae.py --cuda-visible-devices 0 --sequence-length 400 --state-dim 80 --batch-size 512 --epochs 100 --lr 0.003 --lr-decay-factor 0.0001 training_features_file.f32 model_dir_name
```

## Testing
```
./test.sh model01/checkpoints/checkpoint_epoch_50.pth ~/LPCNet/wav/all.wav out_16k.sw
```

# Ideas

1. Not sure about training with tanh clamping at +/-1.  We should be using soft decision information, e.g. 1 symbol plus 1 of noise is a "very likely" 1 symbol.

1. Can we include maximum likelyhood detection in rx side of the bottelneck?  E.g. a +2 received means very likely +1 was transmitted, and shouldn't have the same weight in decoding operation as a 0 received.  Probability of received symbol.

1. Plot scatter diagram of Tx to see where symbols are being mapped.  Would expect random rotations, and magntitudes
   near 1.

1. Reshape pairs of symbols to QPSK, as I think effect of noise will be treated differently in a 2D mapping maybe sqrt(2) better).

1. Reshape into matrix with Nc=number of carriers columns to simulate OFDM.

1. Ability to inject different levels of noise at test time.

1. Using OFDM matrix, simulate symbol by symbol fading channel.  This is a key test.  Need an efficient way to generate fading data, maybe create using Octave for now, an hours worth, or rotate around different channels while training.

1. Confirm SNR calculations, maybe print them, or have SNR3k | Es/No as cmd line options

1. PAPR optimisation.  We could explicitely calculate PAPR, or optimise for maximum average power.  Be interesting to observe envelope of waveform as it trains. We might need to include sync symbols.

1. Way to write/read bottleneck vectors (channel symbols)

1. Look at bottleneck vectors, PCA, any correlation?  Ameniable to VQ?  Further dim reduction? VQ would enable comparative test using classical methods.
