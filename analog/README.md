# Analog Modulation of Vocoder Features

1. Train a basic autoencoder on feature file `train_b20_ml.f32`, model saved to `nn2_cat2.pt`
   ```
   python3 autoencoder1.py ~/Downloads/train_b20_ml.f32 --lr 0.05 --epochs 100 --bottle_dim 20 --ncat 2 --save_model nn2_cat2.pt --noplot
   ```
1. Run inference on feature file `big_dog_b.f32` and enter frame by frame GUI:
   ```
   python3 autoencoder1.py ../manifold/big_dog_b.f32 --bottle_dim 20 --ncat 2 --inference nn2_cat2.pt
   ```
