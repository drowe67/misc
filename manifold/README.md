# Manifold README

Experiments with non-linear Codec 2 dependencies and low F0 speech.

| Files | Description |
| ---- | ---- |
| manifold.tex | Latex write up of experiments |
| ratek_resampler.mk | Makefile to build training data and VQs |
| ratek_resampler.sh | Ties C and Octave code together to automate experiment |
| ratek3_batch.m | Octave batch processing tools |
| ratek3_fbf.m | Experiment 1 (VQ) Octave frame by frame processing/visualisation tool, generates .tex plots |
| manifold.py  | PyTorch ML experiment, training and inference modes |
| ml_fbf.m | Experiment 2 (ML) Octave frame by frame processing/visualisation tool, generates .tex plots |
| sec_order.m | Second order system Octave script to explore problem |
| loss_func.m | Octave script to explore proposed weighted loss function |

# VQ Experiment

1. Build C tools from ~/codec2-dev dr-papr branch, Git hash e430c433bcb34c6:

1. Build K=20 & K=80 M=512 2 stage VQs with small train_120 test database, output PNG VQ plot
   ```
   make -f ratek_resampler.mk
   ```
1. Build K=20 & K=80 M=4096 2 stage VQs with larger train test database, output PNG VQ plot
   ```
   TRAIN=train M=4096 make -f ratek_resampler.mk
   ```
1. After building VQs, run an experiment to generate speech files for listening
   ```
   ./ratek_resampler.sh vq_test_231028
   ```
1. Frame by frame version of K=20:
   ```
   ratek3_fbf("big_dog",165,"train_k20_vq1.f32","train_k20_vq2.f32",20,20)
   ```
1. Frame by frame version of K=80:
   ```
   ratek3_fbf("big_dog",165,"train_k80_vq1.f32","train_k80_vq2.f32",79,100)
   ```
1. Speech sample files for VQ experiment in  ratek_20_80 (see table below)

# ML Experiment

1. Generate ML training data:
   ```
   TRAIN=train make -f ratek_resampler.mk train_b20_ml.f32 train_y80_ml.f32
   ```
1. Run ML training:
   ```
   python3 manifold.py ~/Downloads/train_b20_ml.f32 ~/Downloads/train_y80_ml.f32 --lr 0.2
   ```
1. Run ML inference:
   ```
   ./ratek_resampler.sh ml_test_231126
   ```
1. Speech sample files for ML experiment in 231229_ratek_out (see table below)

# Samples in `vq_20_80`

   | Index | Processing |
   | ---- | ---- |
   | 1 | orginal amplitudes and phase
   | 2 | K=20 mel unquantised, phase0 |
   | 3 | K=20 mel 24 bit two stage VQ, phase0 |
   | 4 | K=40 mel unquantised, phase0 |
   | 5 | K=40 mel 24 bit two stage VQ, phase0 |
   | 6 | K=80 linear unquantised, phase0 |
   | 7 | K=80 linear 24 bit two stage VQ, phase0 |
   | 8 | Codec 2 3200 anchor |

 # Samples in `231229_ratek_out`

   | Index | Processing |
   | ---- | ---- |
   | 1 | orginal amplitudes and phase
   | 2 | phase0, Nb=20 filtering, equivalent to speech synthesised from b vectors, low anchor |
   | 3 | phase0, ideal y vectors, high anchor |
   | 4 | phase0, ideal y vectors, should be the same as 3 (toolchain test) |
   | 5 | phase0, y_hat =F(b) from ML inference  |
   | 6 | same a 5, but refactored ML toolchain into enc/dec functions as cross check |
   | 8 | Codec 2 3200 high anchor |



   
  
