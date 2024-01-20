#!/bin/bash -x
#
# Run some automated autoencoder tests and plot results

epochs=20

if [ $# -gt 0 ]; then
  case $1 in
    clean)
        rm -f loss_*.txt
      ;;
    esac
fi

# Bottleneck same dim as input, should be perfect - but it's not.  Worth looking into.....
if [ ! -f loss_1.txt ]; then
    python3 autoencoder4.py ~/Downloads/train_b20_ml.f32 --lr 0.1 --epochs ${epochs} --bottle_dim 20 --nn 1 --lower_limit_dB 0 --noplot --loss_file loss_1.txt
fi
# zero mean
if [ ! -f loss_2.txt ]; then
    python3 autoencoder4.py ~/Downloads/train_b20_ml.f32 --lr 0.1 --epochs ${epochs} --bottle_dim 20 --nn 1 --lower_limit_dB 0 --noplot --zero_mean --loss_file loss_2.txt
fi
# zero mean, lower loss rate
if [ ! -f loss_3.txt ]; then
    python3 autoencoder4.py ~/Downloads/train_b20_ml.f32 --lr 0.05 --epochs ${epochs} --bottle_dim 20 --nn 1 --lower_limit_dB 0 --noplot --zero_mean --loss_file loss_3.txt
fi
# more complex network
if [ ! -f loss_4.txt ]; then
    python3 autoencoder4.py ~/Downloads/train_b20_ml.f32 --lr 0.1 --epochs ${epochs} --bottle_dim 20 --nn 2 --lower_limit_dB 0 --noplot --zero_mean --loss_file loss_4.txt
fi
# w/o zero mean
if [ ! -f loss_5.txt ]; then
    python3 autoencoder4.py ~/Downloads/train_b20_ml.f32 --lr 0.1 --epochs ${epochs} --bottle_dim 20 --nn 2 --lower_limit_dB 0 --noplot --loss_file loss_5.txt
fi
# smaller network, multiple stages
if [ ! -f loss_6.txt ]; then
    python3 autoencoder4.py ~/Downloads/train_b20_ml.f32 --lr 0.1 --epochs ${epochs} --bottle_dim 20 --nn 3 --lower_limit_dB 0 --noplot --zero_mean --loss_file loss_6.txt
fi
# like #2 but with Adam optimiser
if [ ! -f loss_7.txt ]; then
    python3 autoencoder4.py ~/Downloads/train_b20_ml.f32 --lr 0.00001 --epochs ${epochs} --bottle_dim 20 --nn 1 --lower_limit_dB 0 --noplot --zero_mean --opt Adam --loss_file loss_7.txt
fi
# like #3, but norm, lower loss rate
if [ ! -f loss_8.txt ]; then
    python3 autoencoder4.py ~/Downloads/train_b20_ml.f32 --lr 0.05 --epochs ${epochs} --bottle_dim 20 --nn 1 --lower_limit_dB 0 --noplot --zero_mean --norm --loss_file loss_8.txt
fi

# run same design a few times
if [ ! -f loss_B1.txt ]; then
    python3 autoencoder4.py ~/Downloads/train_b20_ml.f32 --lr 0.1 --epochs ${epochs} --bottle_dim 20 --nn 1 --lower_limit_dB 0 --noplot --loss_file loss_B1.txt
fi
if [ ! -f loss_B2.txt ]; then
    python3 autoencoder4.py ~/Downloads/train_b20_ml.f32 --lr 0.1 --epochs ${epochs} --bottle_dim 20 --nn 1 --lower_limit_dB 0 --noplot --loss_file loss_B2.txt
fi
if [ ! -f loss_B3.txt ]; then
    python3 autoencoder4.py ~/Downloads/train_b20_ml.f32 --lr 0.1 --epochs ${epochs} --bottle_dim 20 --nn 1 --lower_limit_dB 0 --noplot --loss_file loss_B3.txt
fi

# nn5, gradually decreasing towards bottelneck
if [ ! -f loss_B4.txt ]; then
    python3 autoencoder4.py ~/Downloads/train_b20_ml.f32 --lr 0.1 --epochs ${epochs} --bottle_dim 20 --nn 5 --lower_limit_dB 0 --noplot --loss_file loss_B4.txt
fi

# nn1 with dim 15 bottleneck
if [ ! -f loss_B5.txt ]; then
    python3 autoencoder4.py ~/Downloads/train_b20_ml.f32 --lr 0.1 --epochs ${epochs} --bottle_dim 20 --nn 5 --lower_limit_dB 0 --noplot --loss_file loss_B5.txt
fi

# nn1 with dim 10 bottleneck
if [ ! -f loss_B6.txt ]; then
    python3 autoencoder4.py ~/Downloads/train_b20_ml.f32 --lr 0.1 --epochs ${epochs} --bottle_dim 20 --nn 5 --lower_limit_dB 0 --noplot --loss_file loss_B6.txt
fi

