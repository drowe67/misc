# Manifold README

Experiments with non-linear Codec 2 dependencies and low F0 speech.

| Files | Description |
| ---- | ---- |
| manifold.tex | Latex write up of experiments |
| ratek_resampler.mk | Makefile to build training data and VQs |
| ratek_resampler.sh | Shell script to automate various parts of experiment |
| ratek3_batch.m | Octave batch processing tool for linear resampling |

1. Build C tools from ~/codec2-dev dr-papr branch, Git hash e430c433bcb34c6:

1. Build K=20 & K=80 VQs with small train_120 test database, PNG VQ perf plot
   ```
   make -f ratek_resampler.mk
   ```
