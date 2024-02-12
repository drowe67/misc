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
parser.add_argument('features', type=str, help='path to input feature file in .f32 format')
parser.add_argument('features_hat', type=str, help='path to output feature file in .f32 format')
parser.add_argument('--latent-dim', type=int, help="number of symbols produces by encoder, default: 80", default=80)
parser.add_argument('--cuda-visible-devices', type=str, help="comma separates list of cuda visible device indices, default: ''", default="")
parser.add_argument('--write_latent', type=str, default="", help='path to output file of latent vectors z[latent_dim] in .f32 format')
parser.add_argument('--EsNodB', type=float, default=0, help='latent symbol Es/No in dB (note different to QPSK symbol Es/No)')
parser.add_argument('--passthru', action='store_true', help='copy features in to feature out, bypassing ML network')
parser.add_argument('--test_mp', action='store_true', help='Fixed notch test multipath channel')
parser.add_argument('--mp_file', type=str, default="", help='path to multipath file, rate Rs time steps by Nc carriers .f32 format')
args = parser.parse_args()

# set visible devices
os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda_visible_devices

# device
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

latent_dim = args.latent_dim

# not exposed
nb_total_features = 36
num_features = 20
num_used_features = 20

# load model from a checkpoint file
model = RDOVAE(num_features, latent_dim, args.EsNodB)
checkpoint = torch.load(args.model_name, map_location='cpu')
model.load_state_dict(checkpoint['state_dict'], strict=False)
checkpoint['state_dict'] = model.state_dict()

# dataloader
feature_file = args.features
features_in = np.reshape(np.fromfile(feature_file, dtype=np.float32), (1, -1, nb_total_features))
nb_features_rounded = model.dec_stride*(features_in.shape[1]//model.dec_stride)
features = features_in[:,:nb_features_rounded,:]
features = features[:, :, :num_used_features]
features = torch.tensor(features)
print(f"Processing: {nb_features_rounded} feature vectors")

# default multipath model H=1
Rs = model.get_Rs()
Nc = model.get_Nc()
num_timesteps_at_rate_Rs = int((nb_features_rounded // model.get_enc_stride())*model.get_Ns())
H = torch.ones((1,num_timesteps_at_rate_Rs,Nc))

# construct a contrived multipath model, will be a series of peaks an notches, between H=2 an H=0
if args.test_mp:
   G1 = 1
   G2 = 1
   d  = 0.002
   Rs = model.get_Rs()
   Nc = model.get_Nc()

   num_timesteps_at_rate_Rs = int(nb_features_rounded*model.get_Rs()/model.get_Rfeat())
   for c in range(Nc):
      omega = 2*np.pi*c
      arg = torch.tensor(-1j*omega*d*Rs)
      H[0,:,c] = torch.abs(G1 + G2*torch.exp(arg))  # in this case channel doesn't evolve over time
                                                    # only mag matters, we assume external phase equalisation

# user supplied multipath model
if args.mp_file:
   H = np.reshape(np.fromfile(args.mp_file, dtype=np.float32), (1, -1, Nc))
   print(H.shape, num_timesteps_at_rate_Rs)
   if H.shape[1] < num_timesteps_at_rate_Rs:
      print("Multipath file too short")
      quit()
   H = H[:,:num_timesteps_at_rate_Rs,:]
   #hf_gain = np.std(H)
   #H = H/hf_gain
   print(H.shape,np.var(H))
   H = torch.tensor(H)

if __name__ == '__main__':

   if args.passthru:
      features_hat = features_in.flatten()
      features_hat.tofile(args.features_hat)
      quit()

   # push model to device and run test
   model.to(device)
   features.to(device)
   H.to(device)
   output = model(features,H)

   # lets check actual Eq/No, Eb/No and SNR, and monitor assumption |z| ~ 1, especially for multipath
   tx_sym = output["tx_sym"].detach().numpy()
   Eq_meas = np.var(tx_sym)
   No = model.get_sigma()**2
   EqNodB_meas = 10*np.log10(Eq_meas/No)
   Rq = Rs*Nc
   B = 3000
   SNRdB_meas = EqNodB_meas + 10*np.log10(Rq/B)
   print(f"Measured: Eq: {Eq_meas:5.2f} EqNodB: {EqNodB_meas:5.2f} EbNodB: {EqNodB_meas-3:5.2f} SNR3kdB: {SNRdB_meas:5.2f}")

   features_hat = output["features_hat"]
   features_hat = torch.cat([features_hat, torch.zeros_like(features_hat)[:,:,:16]], dim=-1)
   features_hat = features_hat.detach().numpy().flatten().astype('float32')
   features_hat.tofile(args.features_hat)

   if len(args.write_latent):
      z_hat = output["z_hat"].detach().numpy().flatten().astype('float32')
      z_hat.tofile(args.write_latent)
   
