# Analog Modulation and Quantisation of Vocoder Features

Experiments with autoencoders, quantisation, and analog modulation of Vocoder features.  Using Codec 2
features for current work (as I am familiar with them), but could also be applied to other (e.g. neural)
vocoders.

| Files | Description |
| ---- | ---- |
| analog.tex | Latex write up of analog concept |
| mlquant.tex | Latex write up of ML quantisation work |
| analog.sh | Ties C and Octave, and Pyton code together to automate experiments |
| linear_batch.m | Octave batch processing tools that perform linear DSP |
| analog.py  | PyTorch ML part of analog experiment, training, inference, and visualisation modes |
| autoencoder1.py  | PyTorch ML part of ML autoencoder and quantisation experiments, training, inference, and visualisation modes, b->b_hat |
| mlquant_plots.m | Octave scripts to generate plots for mlquant.tex |
| autoencoder4.py  | Pytorch ML to test autoencoder design, e.g. with d=20 bottlenck, should be low distortion |
| auto_test.sh  | Shell script that drives autoencoder4.py in various configurations |
| auto_plots.sh  | Octave script to plot loss curves from auto_test.sh |
| autoencoder2.py | Variation on autoencoder1.py that does y->y_hat |
| autoencoder3.py | Variation on autoencoder1.py that does b->y_hat |

1. Build C tools from ~/codec2-dev dr-papr branch, Git hash c43d2958c6c:

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
 
1. Dim reduction demo (TODO fix this demo, not working as of 200121):
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
   python3 autoencoder1.py ~/Downloads/train_b20_ml.f32 --bottle_dim 10 --ncat 1 --nn 4 --norm --inference nn4_cat1.pt --read_latent train_l_hat.f32
   ```

1. Run VQ-ed autoencoder 4 experiment (Experiment 1 in `mlquant.tex`) using d=10 bottleneck:
   ```
   ./analog.sh test_240118
   ```
   Results are in `240118_wav` (see legend below).  Result was an unacceptable drop in quality on male samples, traced to the bottleneck of the autoencoder step.  VQ seemed to work Ok. Female sample OK.  More in ml_quant.tex/pdf

1. Experiments with gamma, autoencoder3.py (b->y_hat)
   ```
   python3 autoencoder3.py ~/Downloads/train_b20_ml.f32 ~/Downloads/train_y80_ml.f32 --lr 2 --epochs 50 --nn 1 --frame 63  --bottle_dim 10 --gamma 0.9
   ```

# Samples in `240118_wav` (Experiment 1)

   | Index | Processing | Results |
   | ---- | ---- | ---- |
   | 1 | original amplitudes and phase | OK |
   | 3 | phase0, y_hat =F(b) from ML inference  (manifold project) | OK |
   | 5 | As per 3, but K=20 `b` vectors passed through autoencoder 4 d=10 bottleneck, no quantisation | Male drop in quality |
   | 6 | As per 5, 24 bit VQ of bottleneck vectors | Male drop in quality |
   | 7 | As per 5, 12 bit VQ of bottleneck vectors | Male drop in quality |
   | 8 | Codec 2 3200 high anchor | - |

