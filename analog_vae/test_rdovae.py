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
parser.add_argument('output', type=str, help='path to output feature file in .f32 format')
parser.add_argument('--cuda-visible-devices', type=str, help="comma separates list of cuda visible device indices, default: ''", default="")
parser.add_argument('--write_latent', type=str, default="", help='path to output file of latent vectors z[latent_dim] in .f32 format')
parser.add_argument('--EsNodB', type=float, default=0, help='per symbol SNR in dB')
parser.add_argument('--passthru', action='store_true', help='copy features in to feature out, bypassing ML network')
model_group = parser.add_argument_group(title="model parameters")
model_group.add_argument('--latent-dim', type=int, help="number of symbols produces by encoder, default: 80", default=80)
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
checkpoint['state_dict']    = model.state_dict()

# dataloader
feature_file = args.features
features_in = np.reshape(np.fromfile(feature_file, dtype=np.float32), (1, -1, nb_total_features))
nb_features_rounded = model.dec_stride*(features_in.shape[1]//model.dec_stride)
features = features_in[:,:nb_features_rounded,:]
features = features[:, :, :num_used_features]
features = torch.tensor(features)
print(f"Processing: {nb_features_rounded}")

if __name__ == '__main__':

   if args.passthru:
      output = features_in.flatten()
      output.tofile(args.output)
      quit()

   # push model to device and run test
   model.to(device)
   features.to(device)
   output,z,tx_sym = model(features)

   # lets check actual Es/No and assumption |z| ~ 1
   Es_meas = np.var(tx_sym.detach().numpy())
   No = model.get_noise_std()**2
   EsNodB_meas = 10*np.log10(Es_meas/No)
   print(f"Measured EsNodB: {EsNodB_meas:5.2f}")

   output = torch.cat([output, torch.zeros_like(output)[:,:,:16]], dim=-1)
   output = output.detach().numpy().flatten().astype('float32')
   output.tofile(args.output)

   if len(args.write_latent):
      z = z.detach().numpy().flatten().astype('float32')
      z.tofile(args.write_latent)
   