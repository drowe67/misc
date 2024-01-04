# Analog Modulation of Vocoder Features

Experiments with autoencoders, quantisation, and analog modulation of Vocoder features.  Using Codec 2
features for current work (as I am familiar with them), but could also be applied to other (e.g. neural)
vocoders.

| Files | Description |
| ---- | ---- |
| analog.tex | Latex write up of experiments |
| analog.sh | Ties C and Octave, and Pyton code together to automate experiment |
| linear_batch.m | Octave batch processing tools that perform linear DSP |
| analog.py  | PyTorch ML part of experiment, training, inference, and visualisation modes |

1. Build C tools from ~/codec2-dev dr-papr branch, Git hash e430c433bcb34c6:

1. Train a basic autoencoder on feature file `train_b20_ml.f32`, model saved to `nn2_cat2.pt`
   ```
   python3 autoencoder1.py ~/Downloads/train_b20_ml.f32 --nn 2 --lr 0.05 --epochs 100 --bottle_dim 20 --ncat 2 --save_model nn2_cat2.pt --noplot
   ```
   This example concatenates two adjacent feature vectors and has a dimension 20 bottleneck.

1. Run inference on feature file `big_dog_b.f32` and enter frame by frame visualisation GUI:
   ```
   python3 autoencoder1.py ../manifold/big_dog_b.f32 --nn 2 --bottle_dim 20 --ncat 2 --inference nn2_cat2.pt
   ```
