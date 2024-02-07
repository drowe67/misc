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
python3 ./train_rdovae.py --cuda-visible-devices 0 --sequence-length 400 --batch-size 512 --epochs 100 --lr 0.003 --lr-decay-factor 0.0001 training_features_file.f32 model_dir_name
```

## Testing
```
./test.sh model01/checkpoints/checkpoint_epoch_50.pth ~/LPCNet/wav/all.wav out_16k.sw
```

# Notes

1. Issues: We would like smooth degredation from high SNR to low SNR, rather than training and operating at one SNR.  Currently if trained at 10dB, not as good as model trained at 0dB when tested at 0dB.  Also, if trained at 0dB, quality is reduced when tested on high SNR channel, compared to model trained ta high SNR.

1. Test: Multipath with no noise should mean speech is not impaired, as no "symbol errors".

1. Test: co-channel interference, interfering sinusoids, and impulse noise, non-flat passband.

1. Test: level sensitivity, do we need/assume amplitude normalisation?

1. Test: Try vocoder with several speakers and background noise.

1. Not sure about training with tanh clamping at +/-1 on rx, as fading and noise will push it >>1.  We should be using soft decision information, e.g. 1 symbol plus 1 of noise is a "very likely" 1 symbol.

1. Can we include maximum likelyhood detection in rx side of the bottelneck?  E.g. a +2 received means very likely +1 was transmitted, and shouldn't have the same weight in decoding operation as a 0 received.  Probability of received symbol.

1. Plot scatter diagram of Tx to see where symbols are being mapped.

1. ~Reshape pairs of symbols to QPSK, as I think effect of noise will be treated differently in a 2D mapping maybe sqrt(2) better.~

1. ~Reshape into matrix with Nc=number of carriers columns to simulate OFDM.~

1. ~Ability to inject different levels of noise at test time.~

1. Using OFDM matrix, simulate symbol by symbol fading channel.  This is a key test.  Need an efficient way to generate fading data, maybe create using Octave for now, an hours worth, or rotate around different channels while training.

1. ~Confirm SNR calculations, maybe print them, or have SNR3k | Es/No as cmd line options~

1. PAPR optimisation.  If we put a bottleneck on the peak power, the network should optimise for miminal PAPR (maximise RMS power) for a given noise level. Be interesting to observe envelope of waveform as it trains, and the phase of symbols. We might need to include sync symbols.

1.~ Way to write/read bottleneck vectors (channel symbols)~

1. Look at bottleneck vectors, PCA, any correlation?  Ameniable to VQ?  Further dim reduction? VQ would enable comparative test using classical FEC methods.

1. How can we apply interleaving, e.g./ just spread symbol sover a longer modem frame, or let network spread them.

1. Diversity in frequency - classical DSP or with ML in the loop?

1. Sweep different latent dimensions and choose best perf for given SNR.

1. Can we use loss function as an objective measure for comparing different schemes?
