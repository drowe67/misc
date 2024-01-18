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

1. Install [vector-quantize-pytorch](https://github.com/lucidrains/vector-quantize-pytorch)
   ```
   pip install vector-quantize-pytorch
   ```
 
1. Dim reduction demo:
   ```
   python3 autoencoder1.py ~/Downloads/train_b20_ml.f32 --lr 1E-1 --epochs 20 --bottle_dim 10 --ncat 1 --nn 2 --norm --wloss
   ```

1. Dim reduction with noise in the bottleneck, to simulate the effect of a quantiser, and encourage a well behaived latent distribution.
   ```
   python3 autoencoder1.py ~/Downloads/train_b20_ml.f32 --bottle_dim 10 --ncat 1 --nn 4 --norm --wloss --epochs 100 --noise_var 1E-3 --save_model nn4_cat1.pt
   ```
   Noise is also added during inference, pressing the space bar allows you to see the effect of different noise samples.

1. Using model trained in last step, run without noise and dump latent vectors for analysis and external VQ training:
   ```
   python3 autoencoder1.py ~/Downloads/train_b20_ml.f32 --bottle_dim 10 --ncat 1 --nn 4 --norm --inference nn4_cat1.pt --write_latent l.f32
   ```
   Note after run we enter GUI mode, pressing space bar will replot current frame.  This is interesting when noise is added,
   as the effect of different noise vectors can be observed.

1. VQ training on latent vectors:
   ```
   ~/codec2-dev/build_linux/misc/vqtrain train_l.f32 10 4096 vq1.f32 --split -s 0.0001 -r res1.f32 1>vq_train_var1.txt && ~/codec2-dev/build_linux/misc/vqtrain res1.f32 10 4096 vq2.f32 --split -s 0.0001 -r res2.f32 1>vq_train_var2.txt && ~/codec2-dev/build_linux/misc/vqtrain res2.f32 10 4096 vq3.f32 --split -s 0.0001 1>vq_train_var3.txt
   ```
1. VQ some latent vectors:
   ```
   cat train_l.f32 | ~/codec2-dev/build_linux/misc/vq_mbest -k 10 -q vq1.f32,vq2.f32 --mbest 5 > train_l_hat.f32
   ```

1. Inject VQ-ed vectors back into network:
   ```
   python3 autoencoder1.py ~/Downloads/train_b20_ml.f32 --bottle_dim 10 --ncat 1 --nn 4 --norm --inference nn4_cat1a.pt --read_latent train_l_hat.f32
   ``` 
