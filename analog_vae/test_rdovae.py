"""
/* Copyright (c) 2022 Amazon
   Written by Jan Buethe */
/*
   Redistribution and use in source and binary forms, with or without
   modification, are permitted provided that the following conditions
   are met:

   - Redistributions of source code must retain the above copyright
   notice, this list of conditions and the following disclaimer.

   - Redistributions in binary form must reproduce the above copyright
   notice, this list of conditions and the following disclaimer in the
   documentation and/or other materials provided with the distribution.

   THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
   ``AS IS'' AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
   LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
   A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER
   OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
   EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
   PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
   PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
   LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
   NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
   SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/
"""

import os
import argparse

import numpy as np
import torch
import tqdm

from rdovae import RDOVAE, RDOVAEDataset, distortion_loss


parser = argparse.ArgumentParser()

parser.add_argument('model_name', type=str, help='path to model in .pth format')
parser.add_argument('features', type=str, help='path to feature file in .f32 format')
parser.add_argument('output', type=str, help='path to output folder')

parser.add_argument('--cuda-visible-devices', type=str, help="comma separates list of cuda visible device indices, default: ''", default="")


model_group = parser.add_argument_group(title="model parameters")
model_group.add_argument('--latent-dim', type=int, help="number of symbols produces by encoder, default: 80", default=80)
model_group.add_argument('--cond-size', type=int, help="first conditioning size, default: 256", default=256)
model_group.add_argument('--cond-size2', type=int, help="second conditioning size, default: 256", default=256)
model_group.add_argument('--state-dim', type=int, help="dimensionality of transfered state, default: 24", default=24)
model_group.add_argument('--quant-levels', type=int, help="number of quantization levels, default: 16", default=16)
model_group.add_argument('--lambda-min', type=float, help="minimal value for rate lambda, default: 0.0002", default=2e-4)
model_group.add_argument('--lambda-max', type=float, help="maximal value for rate lambda, default: 0.0104", default=0.0104)
model_group.add_argument('--pvq-num-pulses', type=int, help="number of pulses for PVQ, default: 82", default=82)
model_group.add_argument('--state-dropout-rate', type=float, help="state dropout rate, default: 0", default=0.0)

training_group = parser.add_argument_group(title="training parameters")
training_group.add_argument('--batch-size', type=int, help="batch size, default: 32", default=32)
training_group.add_argument('--lr', type=float, help='learning rate, default: 3e-4', default=3e-4)
training_group.add_argument('--epochs', type=int, help='number of training epochs, default: 100', default=100)
training_group.add_argument('--sequence-length', type=int, help='sequence length, needs to be divisible by 4, default: 256', default=256)
training_group.add_argument('--lr-decay-factor', type=float, help='learning rate decay factor, default: 2.5e-5', default=2.5e-5)
training_group.add_argument('--split-mode', type=str, choices=['split', 'random_split'], help='splitting mode for decoder input, default: split', default='split')
training_group.add_argument('--enable-first-frame-loss', action='store_true', default=False, help='enables dedicated distortion loss on first 4 decoder frames')
training_group.add_argument('--initial-checkpoint', type=str, help='initial checkpoint to start training from, default: None', default=None)
training_group.add_argument('--train-decoder-only', action='store_true', help='freeze encoder and statistical model and train decoder only')

args = parser.parse_args()

# set visible devices
os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda_visible_devices


# training parameters
batch_size = args.batch_size
lr = args.lr
epochs = args.epochs
sequence_length = args.sequence_length
lr_decay_factor = args.lr_decay_factor
split_mode = args.split_mode
# not exposed
adam_betas = [0.8, 0.95]
adam_eps = 1e-8
checkpoint = dict()
checkpoint['batch_size'] = batch_size
checkpoint['lr'] = lr
checkpoint['lr_decay_factor'] = lr_decay_factor
checkpoint['split_mode'] = split_mode
checkpoint['epochs'] = epochs
checkpoint['sequence_length'] = sequence_length
checkpoint['adam_betas'] = adam_betas

# logging
log_interval = 10

# device
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# model parameters
cond_size  = args.cond_size
cond_size2 = args.cond_size2
latent_dim = args.latent_dim
quant_levels = args.quant_levels
lambda_min = args.lambda_min
lambda_max = args.lambda_max
state_dim = args.state_dim
# not expsed
nb_total_features = 36
num_features = 20
num_used_features = 20


# training data
feature_file = args.features

# model
checkpoint['model_args']    = (num_features, latent_dim, quant_levels, cond_size, cond_size2)
checkpoint['model_kwargs']  = {'state_dim': state_dim, 'split_mode' : split_mode, 'pvq_num_pulses': args.pvq_num_pulses, 'state_dropout_rate': args.state_dropout_rate}
model = RDOVAE(*checkpoint['model_args'], **checkpoint['model_kwargs'])
checkpoint = torch.load(args.model_name, map_location='cpu')
model.load_state_dict(checkpoint['state_dict'], strict=False)
checkpoint['state_dict']    = model.state_dict()


# dataloader
checkpoint['dataset_args'] = (feature_file, sequence_length, num_features, 36)
checkpoint['dataset_kwargs'] = {'lambda_min': lambda_min, 'lambda_max': lambda_max, 'enc_stride': model.enc_stride, 'quant_levels': quant_levels}

features = np.reshape(np.fromfile(feature_file, dtype=np.float32), (1, -1, nb_total_features))
nb_features_rounded = model.dec_stride*(features.shape[1]//model.dec_stride)
features = features[:,:nb_features_rounded,:]
features = features[:, :, :num_used_features]
features = torch.tensor(features)

if __name__ == '__main__':

    # push model to device
    model.to(device)
    features.to(device)
    #print(features.dtype)
    output = model(features)
    
    output = torch.cat([output, torch.zeros_like(output)[:,:,:16]], dim=-1)
    #print(output.dtype)
    
    output = output.detach().numpy().flatten().astype('float32')
    

    output.tofile(args.output)
