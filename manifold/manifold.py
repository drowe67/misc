"""
Manifold ML experiment - training script.

Train a model to generate low pitched speech with good time domain energy distribution over a
pitch cycle.
"""

import torch
from torch import nn
import numpy as np
import argparse
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

        # b and y vectors are in x_dB = 20*log10(x), scale down to log10(x).  We don't need to scale
        # Wo and voicing (last two floats in feature vector)
        for i in range(self.num_sequences):
            self.features[i,:20] = self.features[i,:20]/20
            self.targets[i,] = self.targets[i,]/20

        # TODO Each vector should have unit (linear) energy
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
args = parser.parse_args()
feature_file = args.features
target_file = args.target

feature_dim = 22
target_dim = 79
sequence_length=1
batch_size = 8

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
            nn.Linear(512, output_dim)
       )

    def forward(self, x):
        y = self.linear_relu_stack(x)
        return y

model = NeuralNetwork(feature_dim, target_dim).to(device)
print(model)

# TODO custom loss function
# criterion to computes the loss between input and target
loss_fn = nn.MSELoss()

# optimizer that will be used to update weights and biases
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

epochs = 1
for epoch in range(epochs):
    running_loss = 0.0
    for batch,(f,y) in enumerate(dataloader):
        #print(y)
        f = f.to(device)
        y = y.to(device)
        y_hat = model(f)
        #print(y_hat.cpu())
        loss = loss_fn(y, y_hat)   
        loss.backward() 
        optimizer.step()
        optimizer.zero_grad()
 
    running_loss += loss.item()
    print(f'Epochs:{epoch + 1:5d} | ' \
          f'Batches per epoch: {batch + 1:3d} | ' \
          f'Loss: {running_loss / (batch + 1):.10f}')

model.eval()
plt.figure(1)
dataloader_infer = torch.utils.data.DataLoader(dataset, batch_size=1)
print("Click on plot to advance, any key to quit")

with torch.no_grad():
    for b,(f,y) in enumerate(dataloader_infer):
        # TODO: must be an easier way to access data at a given index and run on .to(device)
        # Maybe we run on CPU?
        if b >= args.frame:
            f = f.to(device)
            y = y.to(device)
            y_hat = model(f)
            f_plot = f[0,0,].cpu()
            y_plot = y[0,0,].cpu()
            y_hat_plot = y_hat[0,0,].cpu()
            # TODO: compute a distortion metric like SD or MSE (linear)
            plt.clf()
            plt.subplot(211)
            plt.plot(f_plot)
            t = f"f: {b}"
            plt.title(t)
            plt.subplot(212)
            plt.plot(y_plot,'g')
            plt.plot(y_hat_plot,'r')
            plt.show(block=False)
            loop = plt.waitforbuttonpress(0)
            if loop == True:
                quit()
