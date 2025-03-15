"""
Simple demo of training an AM radio (diode) detector:

  python3 ml_amdet.py

"""

import torch
from torch import nn
import numpy as np
import argparse,sys
from matplotlib import pyplot as plt

class aDataset(torch.utils.data.Dataset):
    def __init__(self, n_sams):   

        self.n_sams = n_sams # number of samples in training dataset

        # generate sampled AM signal training data
        fc = 10E3            # carrier freq
        fm = 1E3             # modulation freq
        t = np.arange(n_sams)
        am = (1+np.cos(2*np.pi*fm*t))*np.cos(2*np.pi*fc*t)
        self.am = torch.tensor(am, dtype=torch.float32)

    def __len__(self):
        return self.n_sams

    def __getitem__(self, index):
        x = self.am[index]
        # our ideal detector
        if x > 0:
            y = x
        else:
            y = 0
        # return am_in, det_out sample pair for training
        return x,y
   
parser = argparse.ArgumentParser()
parser.add_argument('--n_sams', type=int, default=1000, help='number of samples to train with')
parser.add_argument('--epochs', type=int, default=3, help='number of training epochs')
parser.add_argument('--lr', type=float, default=5E-2, help='learning rate')
args = parser.parse_args()

n_sams = args.n_sams
device = "cpu"

dataset = aDataset(n_sams)
dataloader = torch.utils.data.DataLoader(dataset)

class Detector(nn.Module):
    def __init__(self):
        super().__init__()
        self.stack = nn.Sequential(
            nn.Linear(1, 1),
            nn.ReLU(),
       )

    def forward(self, x):
        return self.stack(x)

# TODO: can we lose to(device) ?
model = Detector().to(device)
print(model)
loss_fn = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)

# Train model
for epoch in range(args.epochs):
    sum_loss = 0.0
    for batch,(am_in,det_out) in enumerate(dataloader):
        am_in = am_in.to(device)
        #print(am_in.dtype, det_out.dtype)
        det_out_ = model(am_in)
        loss = loss_fn(det_out, det_out_)
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
"""
model.eval()
model.sigma=0
bits_in = torch.sign(torch.rand(n_syms, np, bps)-0.5)
with torch.no_grad():
    symbols, bits_out = model(bits_in.to(device))
symbols = symbols.cpu().numpy()
print(symbols.shape)
"""
