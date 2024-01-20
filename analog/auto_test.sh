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
    python3 autoencoder4.py ~/Downloads/train_b20_ml.f32 --lr 0.1 --epochs ${epochs} --bottle_dim 20 --nn 1 --lower_limit_dB 0 --noplot --zero_mean --loss_file loss_B1.txt
fi
if [ ! -f loss_B2.txt ]; then
    python3 autoencoder4.py ~/Downloads/train_b20_ml.f32 --lr 0.1 --epochs ${epochs} --bottle_dim 20 --nn 1 --lower_limit_dB 0 --noplot --zero_mean --loss_file loss_B2.txt
fi
if [ ! -f loss_B3.txt ]; then
    python3 autoencoder4.py ~/Downloads/train_b20_ml.f32 --lr 0.1 --epochs ${epochs} --bottle_dim 20 --nn 1 --lower_limit_dB 0 --noplot --zero_mean --loss_file loss_B3.txt
fi

# nn4, gradually decreasing towards bottleneck, d=20
if [ ! -f loss_B4.txt ]; then
    python3 autoencoder4.py ~/Downloads/train_b20_ml.f32 --lr 0.1 --epochs ${epochs} --bottle_dim 20 --nn 4 --lower_limit_dB 0 --noplot --zero_mean --loss_file loss_B4.txt
fi

# nn1 with dim 15 bottleneck
if [ ! -f loss_B5.txt ]; then
    python3 autoencoder4.py ~/Downloads/train_b20_ml.f32 --lr 0.1 --epochs ${epochs} --bottle_dim 15 --nn 1 --lower_limit_dB 0 --noplot --zero_mean --loss_file loss_B5.txt
fi

# nn1 with dim 10 bottleneck
if [ ! -f loss_B6.txt ]; then
    python3 autoencoder4.py ~/Downloads/train_b20_ml.f32 --lr 0.1 --epochs ${epochs} --bottle_dim 10 --nn 1 --lower_limit_dB 0 --noplot --zero_mean --loss_file loss_B6.txt
fi

# nn4, gradually decreasing towards bottelneck, d=15
if [ ! -f loss_B7.txt ]; then
    python3 autoencoder4.py ~/Downloads/train_b20_ml.f32 --lr 0.1 --epochs ${epochs} --bottle_dim 15 --nn 4 --lower_limit_dB 0 --noplot --zero_mean --loss_file loss_B7.txt
fi

# nn4, gradually decreasing towards bottelneck, d=15
if [ ! -f loss_B8.txt ]; then
    python3 autoencoder4.py ~/Downloads/train_b20_ml.f32 --lr 0.1 --epochs ${epochs} --bottle_dim 15 --nn 4 --lower_limit_dB 0 --noplot --zero_mean --loss_file loss_B8.txt
fi

# nn4, gradually decreasing towards bottelneck, d=15
if [ ! -f loss_B9.txt ]; then
    python3 autoencoder4.py ~/Downloads/train_b20_ml.f32 --lr 0.1 --epochs ${epochs} --bottle_dim 15 --nn 4 --lower_limit_dB 0 --noplot --zero_mean --loss_file loss_B9.txt
fi

# nn5, gradually decreasing towards bottelneck, tanh d=15
if [ ! -f loss_B10.txt ]; then
    python3 autoencoder4.py ~/Downloads/train_b20_ml.f32 --lr 0.1 --epochs ${epochs} --bottle_dim 15 --nn 5 --lower_limit_dB 0 --noplot --zero_mean --loss_file loss_B10.txt
fi

# --------------------------------------
# Bottleneck experiments - note lower limit increased

# nn1 with dim 15 bottleneck as control (simple network, without noise injection)
if [ ! -f loss_C1.txt ]; then
    python3 autoencoder4.py ~/Downloads/train_b20_ml.f32 --lr 0.1 --epochs ${epochs} --bottle_dim 15 --nn 1 --lower_limit_dB 10 --noplot --zero_mean --loss_file loss_C1.txt
fi
# nn5, tanh d=15, 1E-3 noise
if [ ! -f loss_C2.txt ]; then
    python3 autoencoder4.py ~/Downloads/train_b20_ml.f32 --lr 0.1 --epochs ${epochs} --bottle_dim 15 --nn 5 --lower_limit_dB 10 --noplot --zero_mean --noise_var 1E-3 --loss_file loss_C2.txt
fi
# Repeat of C2
if [ ! -f loss_C3.txt ]; then
    python3 autoencoder4.py ~/Downloads/train_b20_ml.f32 --lr 0.1 --epochs ${epochs} --bottle_dim 15 --nn 5 --lower_limit_dB 10 --noplot --zero_mean --noise_var 1E-3 --loss_file loss_C3.txt
fi

# nn5, tanh d=10, 1E-3 noise
if [ ! -f loss_C4.txt ]; then
    python3 autoencoder4.py ~/Downloads/train_b20_ml.f32 --lr 0.1 --epochs ${epochs} --bottle_dim 10 --nn 5 --lower_limit_dB 10 --noplot --zero_mean --noise_var 1E-3 --loss_file loss_C4.txt
fi
# Repeat of C4
if [ ! -f loss_C5.txt ]; then
    python3 autoencoder4.py ~/Downloads/train_b20_ml.f32 --lr 0.1 --epochs ${epochs} --bottle_dim 10 --nn 5 --lower_limit_dB 10 --noplot --zero_mean --noise_var 1E-3 --loss_file loss_C5.txt
fi

# nn5, tanh d=20, 1E-3 noise
if [ ! -f loss_C6.txt ]; then
    python3 autoencoder4.py ~/Downloads/train_b20_ml.f32 --lr 0.1 --epochs ${epochs} --bottle_dim 20 --nn 5 --lower_limit_dB 10 --noplot --zero_mean --noise_var 1E-3 --loss_file loss_C6.txt
fi

# nn5, tanh d=20, 3E-3 noise
if [ ! -f loss_C7.txt ]; then
    python3 autoencoder4.py ~/Downloads/train_b20_ml.f32 --lr 0.1 --epochs ${epochs} --bottle_dim 20 --nn 5 --lower_limit_dB 10 --noplot --zero_mean --noise_var 3E-3 --loss_file loss_C7.txt
fi

# --------------------------------------
# DCT experiments, lets give the NN some help

# nn1 with dim 20 bottleneck as control (simple network, without noise injection)
if [ ! -f loss_D1.txt ]; then
    python3 autoencoder4.py ~/Downloads/train_b20_ml.f32 --lr 0.1 --epochs ${epochs} --bottle_dim 20 --nn 1 \
     --lower_limit_dB 10 --noplot --zero_mean --dct --loss_file loss_D1.txt
fi

# dim 10
if [ ! -f loss_D2.txt ]; then
    python3 autoencoder4.py ~/Downloads/train_b20_ml.f32 --lr 0.1 --epochs ${epochs} --bottle_dim 10 --nn 1 \
     --lower_limit_dB 10 --noplot --zero_mean --dct --loss_file loss_D2.txt
fi

# nn5, 1E-3 noise, dim 20
if [ ! -f loss_D3.txt ]; then
    python3 autoencoder4.py ~/Downloads/train_b20_ml.f32 --lr 0.1 --epochs ${epochs} --bottle_dim 20 --nn 5 \
     --lower_limit_dB 10 --noplot --zero_mean --dct --noise_var 1E-3 --loss_file loss_D3.txt
fi

# dim 10
if [ ! -f loss_D4.txt ]; then
    python3 autoencoder4.py ~/Downloads/train_b20_ml.f32 --lr 0.1 --epochs ${epochs} --bottle_dim 10 --nn 5 \
     --lower_limit_dB 10 --noplot --zero_mean --dct --noise_var 1E-3 --loss_file loss_D4.txt
fi

# dim 10
if [ ! -f loss_D5.txt ]; then
    python3 autoencoder4.py ~/Downloads/train_b20_ml.f32 --lr 0.1 --epochs ${epochs} --bottle_dim 10 --nn 5 \
     --lower_limit_dB 10 --noplot --zero_mean --dct --noise_var 1E-3 --norm --loss_file loss_D5.txt
fi
