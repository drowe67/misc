"""
ML experiment in training PSK constellations, to support RADAE dimension reduction.

For example, to train a constellation with 2 bits/symbol:

  python3 ml_const.py 2

"""

import torch
from torch import nn
import numpy as np
import argparse,sys
from matplotlib import pyplot as plt
import torch.nn.functional as F

class aDataset(torch.utils.data.Dataset):
    def __init__(self,
                 bps,       # bits per symbol
                 n_syms):   # number of symbols

        self.bps = bps
        self.n_syms = n_syms
        self.bits = torch.sign(torch.rand(n_syms, bps)-0.5)

    def __len__(self):
        return self.n_syms

    def __getitem__(self, index):
        return self.bits[index,:]
   
parser = argparse.ArgumentParser()
parser.add_argument('bps', type=int, help='bits per symbol')
parser.add_argument('--n_syms', type=int, default=10000, help='number of symbols to train with')
parser.add_argument('--EsNodB', type=float, default=10, help='energy per symbol over spectral noise desnity in dB')
parser.add_argument('--epochs', type=int, default=10, help='number of trarining epochs')
parser.add_argument('--lr', type=float, default=5E-2, help='learning rate')
args = parser.parse_args()
bps = args.bps
n_syms = args.n_syms
EsNodB = args.EsNodB
batch_size = 32

dataset = aDataset(bps, n_syms)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)

# Get cpu, gpu or mps device for training.
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")

w1 = 100

class PSKencoder(nn.Module):
    def __init__(self, bps):
        super().__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(bps, w1),
            nn.ReLU(),
            nn.Linear(w1, w1),
            nn.ReLU(),
            nn.Linear(w1, 2)
       )

    def forward(self, bits_in):
        symbol = self.linear_relu_stack(bits_in)
        return symbol

class PSKdecoder(nn.Module):
    def __init__(self, bps):
        super().__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(2, w1),
            nn.ReLU(),
            nn.Linear(w1, w1),
            nn.ReLU(),
            nn.Linear(w1, bps)
       )

    def forward(self, symbol_in):
        bits_out = self.linear_relu_stack(symbol_in)
        return bits_out

class PSKconstellation(nn.Module):
    def __init__(self, bps, EsNodB):
        super().__init__()
        self.bps = bps
        self.encoder = PSKencoder(bps)
        self.decoder = PSKdecoder(bps)
        self.sigma = 10**(-0.5*EsNodB/10)

    def forward(self, bits_in):
        symbol_rect = self.encoder(bits_in)
        symbol = symbol_rect[:,0] + 1j*symbol_rect[:,1]
 
        # limit power of complex symbol
        symbol = torch.tanh(torch.abs(symbol))*torch.exp(1j*torch.angle(symbol))
        # AWGN channel
        symbol = symbol + self.sigma*torch.randn_like(symbol)

        symbol_rect[:,0] = symbol.real
        symbol_rect[:,1] = symbol.imag
        bits_out = self.decoder(symbol_rect)
        #print(symbol_rect)
        #print(bits_out)
        #quit()

        return symbol, bits_out


model = PSKconstellation(bps,EsNodB).to(device)
print(model)
loss_fn = nn.MSELoss(reduction='sum')
optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)

# Train model
for epoch in range(args.epochs):
    sum_loss = 0.0
    for batch,(bits_in) in enumerate(dataloader):
        #print(bits_in)
        bits_in = bits_in.to(device)
        symbol, bits_out = model(bits_in)
        #print(bits_out)
        #quit()
        loss = loss_fn(bits_in, bits_out)
        loss.backward() 
        optimizer.step()
        optimizer.zero_grad()
        if np.isnan(loss.item()):
            print("NAN encountered - quitting (try reducing lr)!")
            quit()
        sum_loss += loss.item()

    print(f'Epochs:{epoch + 1:5d} | ' \
        f'Batches per epoch: {batch + 1:3d} | ' \
        f'Loss: {sum_loss / (batch + 1):.10f}')

# Inference using trained model
model.eval()
model.sigma=0
bits_in = torch.sign(torch.rand(n_syms, bps)-0.5)
with torch.no_grad():
    symbols, bits_out = model(bits_in.to(device))
symbols = symbols.cpu().numpy()
print(symbols[:10])
plt.plot(symbols.real,symbols.imag,'+')
plt.axis([-2,2,-2,2])
plt.show()

