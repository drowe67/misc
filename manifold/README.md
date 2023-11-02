# Manifold README

Experiments with non-linear Codec 2 dependencies and low F0 speech.

| Files | Description |
| ---- | ---- |
| manifold.tex | Latex write up of experiments |
| ratek_resampler.mk | Makefile to build training data and VQs |
| ratek_resampler.sh | Ties C and Octave code together to automate experiment |
| ratek3_batch.m | Octave batch processing tool |
| ratek3_fbf.m | Octave frame by frame processing/visualisation tool |

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
   ./ratek_resampler.sh vq_test_23102
   ```
1. Frame by frame version of K=20:
   ```
   ratek3_fbf("big_dog",165,"train_k20_vq1.f32","train_k20_vq2.f32",20,20)
   ```
1. Frame by frame version of K=80:
   ```
   ratek3_fbf("big_dog",165,"train_k80_vq1.f32","train_k80_vq2.f32",79,100)
   ```

   # Samples in `vq_20_80`

   | Index | Processing |
   | ---- | ---- |
   | 1 | orginal amplitudes and phase
   | 2 | K=20 mel unquantised, synthetic phases |
   | 3 | K=20 mel 24 bit two stage VQ, synthetic phases |
   | 4 | K=40 mel unquantised, synthetic phases |
   | 5 | K=40 mel 24 bit two stage VQ, synthetic phases |
   | 6 | K=80 linear unquantised, synthetic phases |
   | 7 | K=80 linear 24 bit two stage VQ, synthetic phases |
   | 8 | Codec 2 3200 anchor |


   
