"""
ML experiment in training PSK constellations, to support RADAE dimension reduction.

For example, to train a 16-QAM constellation (4 bits/symbol):

  python3 ml_const.py 4 --lr 0.001 --EsNodB 20 --epochs 100

  Two parallel QPSK carriers with one network:

  python3 ml_const.py 2 --lr 0.001 --EsNodB 10 --epochs 10 --np 2 
"""

import torch
from torch import nn
import numpy
import argparse,sys
from matplotlib import pyplot as plt
import torch.nn.functional as F

class aDataset(torch.utils.data.Dataset):
    def __init__(self,
                bps,       # bits per symbol
                np,        # number of symbols processed in parallel
                n_syms):   # number of symbols

        self.bps = bps
        self.n_syms = n_syms
        self.bits = torch.sign(torch.rand(n_syms, np, bps)-0.5)

    def __len__(self):
        return self.n_syms

    def __getitem__(self, index):
        return self.bits[index,:]
   
parser = argparse.ArgumentParser()
parser.add_argument('bps', type=int, help='bits per symbol')
parser.add_argument('--np', type=int, default=1, help='number of symbols processed in parallel')
parser.add_argument('--n_syms', type=int, default=10000, help='number of symbols to train with')
parser.add_argument('--EsNodB', type=float, default=10, help='energy per symbol over spectral noise desnity in dB')
parser.add_argument('--epochs', type=int, default=10, help='number of trarining epochs')
parser.add_argument('--lr', type=float, default=5E-2, help='learning rate')
args = parser.parse_args()
bps = args.bps
np = args.np
n_syms = args.n_syms
EsNodB = args.EsNodB
batch_size = 32

dataset = aDataset(bps, np, n_syms)
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

# maps input bits to np complex symbols at bps bits/symbol
class PSKencoder(nn.Module):
    def __init__(self, np, bps):
        super().__init__()
        self.bps = bps
        self.np = np
        self.bits_dim = np*bps
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(self.bits_dim, w1),
            nn.ReLU(),
            nn.Linear(w1, w1),
            nn.ReLU(),
            nn.Linear(w1, np*2)
       )

    def forward(self, bits_in):
        bits_in = torch.reshape(bits_in,(-1,self.bits_dim))
        symbols = self.linear_relu_stack(bits_in)
        symbols = torch.reshape(symbols,(-1,self.np,2))
        return symbols

class PSKdecoder(nn.Module):
    def __init__(self, np, bps):
        super().__init__()
        self.bps = bps
        self.np = np
        self.bits_dim = np*bps
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(np*2, w1),
            nn.ReLU(),
            nn.Linear(w1, w1),
            nn.ReLU(),
            nn.Linear(w1, self.bits_dim)
       )

    def forward(self, symbols_in):
        symbols_in = torch.reshape(symbols_in,(-1,self.np*2))
        bits_out = self.linear_relu_stack(symbols_in)
        bits_out = torch.reshape(bits_out,(-1,self.np,self.bps))
        return bits_out

class PSKconstellation(nn.Module):
    def __init__(self, bps, np, EsNodB):
        super().__init__()
        self.bps = bps
        self.np = np
        self.encoder = PSKencoder(np,bps)
        self.decoder = PSKdecoder(np,bps)
        self.sigma = 10**(-0.5*EsNodB/10)

    def forward(self, bits_in):
        symbols_rect = self.encoder(bits_in)
        symbols = symbols_rect[:,:,0] + 1j*symbols_rect[:,:,1]
 
        # limit power of complex symbol
        symbols = torch.tanh(torch.abs(symbols))*torch.exp(1j*torch.angle(symbols))
        # AWGN channel
        symbols = symbols + self.sigma*torch.randn_like(symbols)

        symbols_rect[:,:,0] = symbols.real
        symbols_rect[:,:,1] = symbols.imag
        bits_out = self.decoder(symbols_rect)
        return symbols, bits_out


model = PSKconstellation(bps,np,EsNodB).to(device)
print(model)
loss_fn = nn.MSELoss(reduction='sum')
optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)

# Train model
for epoch in range(args.epochs):
    sum_loss = 0.0
    for batch,(bits_in) in enumerate(dataloader):
        bits_in = bits_in.to(device)
        symbol, bits_out = model(bits_in)
        loss = loss_fn(bits_in, bits_out)
        loss.backward() 
        optimizer.step()
        optimizer.zero_grad()
        if numpy.isnan(loss.item()):
            print("NAN encountered - quitting (try reducing lr)!")
            quit()
        sum_loss += loss.item()

    print(f'Epochs:{epoch + 1:5d} | ' \
        f'Batches per epoch: {batch + 1:3d} | ' \
        f'Loss: {sum_loss / (batch + 1):.10f}')

# Inference using trained model
model.eval()
model.sigma=0
bits_in = torch.sign(torch.rand(n_syms, np, bps)-0.5)
with torch.no_grad():
    symbols, bits_out = model(bits_in.to(device))
symbols = symbols.cpu().numpy()
print(symbols.shape)
plt.plot(symbols.real,symbols.imag,'+')
#plt.plot(symbols[:,1].real,symbols[:,1].imag,'+')
#plt.plot(symbols[:,0].real,symbols[:,0].imag,'+')
#plt.plot(symbols[:,1].real,symbols[:,1].imag,'+')
plt.axis([-2,2,-2,2])
plt.show()

