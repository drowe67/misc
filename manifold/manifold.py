"""
Manifold ML experiment - training script.

Train a model to generate low pitched speech with good time domain energy distribution over a
pitch cycle.

make -f ratek_resampler.mk train_120_b20_ml.f32 train_120_y80_ml.f32
python3 manifold.py train_120_b20_ml.f32 train_120_y80_ml.f32 --noplot

"""

import torch
from torch import nn
import numpy as np
import argparse,sys
from matplotlib import pyplot as plt

class f32Dataset(torch.utils.data.Dataset):
    def __init__(self,
                feature_file,
                target_file,
                sequence_length, 
                features_dim,
                target_dim):

        self.sequence_length = sequence_length

        self.features = np.reshape(np.fromfile(feature_file, dtype=np.float32), (-1, features_dim))
        self.targets = np.reshape(np.fromfile(target_file, dtype=np.float32), (-1, target_dim))
        self.num_sequences = self.features.shape[0]

        print(f"num_sequences: {self.num_sequences}")
        assert(self.features.shape[0] == self.targets.shape[0])

        # TODO combine energy norm and /20  steps

        for i in range(self.num_sequences):
            # normalise energy in each vector to 1.0: (a) we are interested NN matching shape, not gain (b)
            # keeps each loss on a similar scale to help gradients (c) a gain difference has a large
            # impact on loss
            e = np.sum(10**(self.features[i,:20]/10))
            edB_feature = 10*np.log10(e)
            self.features[i,:20] -= edB_feature
            e = np.sum(10**(self.targets[i,]/10))
            edB_target = 10*np.log10(e)
            self.targets[i,] -= edB_target

            # b and y vectors are in x_dB = 20*log10(x), scale down to log10(x).  We don't need to scale
            # Wo and voicing (last two floats in feature vector)
            self.features[i,:20] = self.features[i,:20]/20
            self.targets[i,] = self.targets[i,]/20

        # TODO remove low energy vectors so we don't train on noise

    def __len__(self):
        return self.num_sequences

    def __getitem__(self, index):
        features = self.features[index * self.sequence_length: (index + 1) * self.sequence_length, :]
        targets = self.targets[index * self.sequence_length: (index + 1) * self.sequence_length, :]
        return features, targets
    
parser = argparse.ArgumentParser()
parser.add_argument('features', type=str, help='path to feature file [b[22] Wo v] .f32 format')
parser.add_argument('target', type=str, help='path to target file [y[79]] in .f32 format')
parser.add_argument('--frame', type=int, default=165, help='frames to start veiwing')
parser.add_argument('--noplot', action='store_true', help='disable plots after training')
args = parser.parse_args()
feature_file = args.features
target_file = args.target

feature_dim = 22
target_dim = 79
sequence_length=1
batch_size = 32

dataset = f32Dataset(feature_file, target_file, sequence_length, feature_dim, target_dim)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)

for f,y in dataloader:
    print(f"Shape of features: {f.shape} targets: {y.shape}")
    break

# Get cpu, gpu or mps device for training.
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")

# Define model
class NeuralNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, output_dim)
       )

    def forward(self, x):
        y = self.linear_relu_stack(x)
        return y

model = NeuralNetwork(feature_dim, target_dim).to(device)
print(model,sum(p.numel() for p in model.parameters()))

# prototype custom loss function
def my_loss_mse(y_hat, y):
    loss = torch.mean((y_hat - y)**2)
    return loss
 
gamma = 0.5
 
# custom loss function that operates in the weighted linear domain
def my_loss(y_hat, y):
    ten = 10*torch.ones(y.shape)
    ten = ten.to(device)
    y_lin = torch.pow(ten,y)
    y_hat_lin = torch.pow(ten,y_hat)
    w = -(1-gamma)*y
    w = torch.clamp(w,max=30)
    w_lin = torch.pow(ten,w)
    weighted_error = (y_hat_lin - y_lin)*w_lin
    loss = torch.mean(weighted_error**2)
    #print(loss)
    return loss

# test for our custom loss function
x = np.ones(2)
y = 2*np.ones(2)
w = 10**(-(1-gamma)*y)
w = np.clip(w,None,30)
result = my_loss(torch.from_numpy(x).to(device),torch.from_numpy(y).to(device)).cpu()
expected_result = np.mean(((10**x - 10**y)*w)**2)
if np.abs(result - expected_result) > expected_result*1E-3:
    print("my_loss() test: fail")
    print(f"my_loss(): {result} expected: {expected_result}")
    quit()
else:
    print("my_loss() test: pass ")
loss_fn = my_loss

# optimizer that will be used to update weights and biases
optimizer = torch.optim.SGD(model.parameters(), lr=5E-2)

epochs = 100
for epoch in range(epochs):
    sum_loss = 0.0
    for batch,(f,y) in enumerate(dataloader):
        f = f.to(device)
        y = y.to(device)
        y_hat = model(f)
        loss = loss_fn(y, y_hat)
        loss.backward() 
        optimizer.step()
        optimizer.zero_grad()
        if np.isnan(loss.item()):
            quit()
        sum_loss += loss.item()

    print(f'Epochs:{epoch + 1:5d} | ' \
          f'Batches per epoch: {batch + 1:3d} | ' \
          f'Loss: {sum_loss / (batch + 1):.10f}')

if args.noplot:
    quit()

model.eval()

print("[click or n]-next [b]-back [q]-quit")

b_f_kHz = np.array([0.1998, 0.2782, 0.3635, 0.4561, 0.5569, 0.6664, 0.7855, 0.9149, 1.0556, 1.2086, 1.3749, 1.5557,
1.7523, 1.9659, 2.1982, 2.4508, 2.7253, 3.0238, 3.3483, 3.7011])
Fs = 8000
Lhigh = 80
F0high = (Fs/2)/Lhigh
y_f_kHz = np.arange(F0high,Lhigh*F0high, F0high)/1000

# called when we press a key on the plot
akey = ''
def on_press(event):
    global akey
    akey = event.key

plt.figure(1)
fig, ax = plt.subplots()
fig.canvas.mpl_connect('key_press_event', on_press)

with torch.no_grad():
    b = args.frame
    loop = True
    while loop:
        (f,y) = dataset.__getitem__(b)
        y_hat = model(torch.from_numpy(f))
        f_plot = 20*f[0,]
        y_plot = 20*y[0,]
        y_hat_plot = 20*y_hat[0,].cpu()
        # TODO: compute a distortion metric like SD or MSE (linear)
        plt.clf()
        plt.plot(b_f_kHz,f_plot[0:20])
        t = f"f: {b}"
        plt.title(t)
        plt.plot(y_f_kHz,y_plot,'g')
        plt.plot(y_f_kHz,y_hat_plot,'r')
        plt.axis([0, 4, -60, 0])
        plt.show(block=False)
        plt.pause(0.01)
        button = plt.waitforbuttonpress(0)
        if akey == 'b':
            b -= 1
        if akey == 'n' or button == False:
            b += 1
        if akey == 'q':
            loop = False
