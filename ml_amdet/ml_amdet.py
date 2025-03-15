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
        Fs = 100E3           # sample rate
        fc = 10E3            # carrier freq
        fm = 1E3             # modulation freq
        t = np.arange(n_sams)
        am = (1+np.cos(2*np.pi*fm*t/Fs))*np.cos(2*np.pi*fc*t/Fs)
        self.am = torch.tensor(am, dtype=torch.float32)

    def __len__(self):
        return self.n_sams

    def __getitem__(self, index):
        x = self.am[index]
        # our ideal detector
        if x > 0:
            y = x
        else:
            y = torch.zeros_like(x)
        # return am_in, det_out sample pair for training
        #print(x.dtype,y.dtype)
        return x,y
   
parser = argparse.ArgumentParser()
parser.add_argument('--n_sams', type=int, default=1000, help='number of samples to train with')
parser.add_argument('--epochs', type=int, default=1, help='number of training epochs')
parser.add_argument('--lr', type=float, default=0.01, help='learning rate')
args = parser.parse_args()
n_sams = args.n_sams

dataset = aDataset(n_sams)
dataloader = torch.utils.data.DataLoader(dataset)

# Our network is a 1 element linear layer followed by a ReLU() non-linear layer
class Detector(nn.Module):
    def __init__(self):
        super().__init__()
        self.stack = nn.Sequential(
            nn.Linear(1, 1),
            nn.ReLU(),
        )
        # init weight & bias to known value to get repeatable results
        # (not usually necessary, but useful for our demo)
        with torch.no_grad():
            self.stack[0].weight.copy_((0.25))
            self.stack[0].bias.copy_((0.25))

    def forward(self, x):
        return self.stack(x)

model = Detector()
print(model)

    
loss_fn = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)
print(model.stack[0].weight, model.stack[0].bias)

# Train model
for epoch in range(args.epochs):
    am_in_log = []
    det_out_log = []
    det_out__log = []
    sum_loss = 0.0
    for batch,(am_in,det_out) in enumerate(dataloader):
        det_out_ = model(am_in)
        loss = loss_fn(det_out, det_out_)
        loss.backward() 
        optimizer.step()
        optimizer.zero_grad()
        if np.isnan(loss.item()):
            print("NAN encountered - quitting (try reducing lr)!")
            quit()
        sum_loss += loss.item()

        # log variables for plotting
        am_in_log = np.concatenate((am_in_log,am_in));
        det_out_log = np.concatenate((det_out_log,det_out));
        det_out__log = np.concatenate((det_out__log,det_out_.detach().numpy()));

    plt.subplot(311)
    plt.plot(am_in_log)
    plt.ylabel('AM In')
    plt.subplot(312)
    plt.plot(det_out_log);
    plt.ylabel('Det Out')
    plt.subplot(313)
    plt.plot(det_out__log,'r');
    plt.ylabel('Det Out ML')
    plt.show()

    print(f'Epochs:{epoch + 1:5d} | ' \
        f'Batches per epoch: {batch + 1:3d} | ' \
        f'Loss: {sum_loss / (batch + 1):.10f}')
    
    print(model.stack[0].weight, model.stack[0].bias)

