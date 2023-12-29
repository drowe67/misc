"""
Toy autoencoder to gain experience with PyTorch
"""

import torch
from torch import nn
import numpy as np
import argparse

# from: https://github.com/drowe67/opus/blob/dr-minor-typos/dnn/torch/rdovae/rdovae/dataset.py
# Opus GitHub repo opus-ng branch

class LPCNetDataset(torch.utils.data.Dataset):
    def __init__(self,
                feature_file,
                sequence_length,
                num_used_features=20,
                num_features=36):

        self.sequence_length = sequence_length
 
        self.features = np.reshape(np.fromfile(feature_file, dtype=np.float32), (-1, num_features))
        self.features = self.features[:, :num_used_features]
        self.num_sequences = self.features.shape[0] // sequence_length

    def __len__(self):
        return self.num_sequences

    def __getitem__(self, index):
        features = self.features[index * self.sequence_length: (index + 1) * self.sequence_length, :]
        return features
    
parser = argparse.ArgumentParser()
parser.add_argument('features', type=str, help='path to feature file in .f32 format')
args = parser.parse_args()

feature_file = args.features
num_used_features = 20
num_features = 22
sequence_length = 1
batch_size = 32

dataset = LPCNetDataset(feature_file, sequence_length, num_used_features, num_features)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)

for f in dataloader:
    print(f"Shape of features: {f.shape}")
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
    def __init__(self, input_dim):
        super().__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, input_dim)
       )

    def forward(self, x):
        y = self.linear_relu_stack(x)
        return y

model = NeuralNetwork(num_used_features).to(device)
print(model)

# criterion to computes the loss between input and target
loss_fn = nn.MSELoss()

# optimizer that will be used to update weights and biases
optimizer = torch.optim.SGD(model.parameters(), lr=0.0001)

epochs = 10
for epoch in range(epochs):
    running_loss = 0.0
    for batch, x in enumerate(dataloader):
        x = x.to(device)
        y = model(x)
        loss = loss_fn(x, y)   
        loss.backward() 
        optimizer.step()
        optimizer.zero_grad()
 
    running_loss += loss.item()
    print(f'Epochs:{epoch + 1:5d} | ' \
          f'Batches per epoch: {batch + 1:3d} | ' \
          f'Loss: {running_loss / (batch + 1):.10f}')
