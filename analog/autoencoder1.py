"""
Autoencoder attempt 1, to gain experience in concatenating multiple frames of features
"""

import torch
from torch import nn
import numpy as np
import argparse

# loading datasets in .f32 files
class f32Dataset(torch.utils.data.Dataset):
    def __init__(self,
                feature_file,
                sequence_length,
                num_used_features=20,
                num_features=22):

        self.sequence_length = sequence_length
 
        self.features = np.reshape(np.fromfile(feature_file, dtype=np.float32), (-1, num_features))
        self.features = self.features[:, :num_used_features]
        self.num_sequences = self.features.shape[0] - sequence_length + 1

    def __len__(self):
        return self.num_sequences

    # overlapping sequences to make better use of training data
    def __getitem__(self, index):
        #features = self.features[index * self.sequence_length: (index + 1) * self.sequence_length, :]
        features = self.features[index: (index + self.sequence_length), :]
        return features
    
parser = argparse.ArgumentParser()
parser.add_argument('features', type=str, help='path to feature file in .f32 format')
parser.add_argument('--lr', type=float, default=5E-2, help='learning rate')
parser.add_argument('--bottle_dim', type=int, default=10, help='bottleneck dim')
parser.add_argument('--epochs', type=int, default=10, help='number of training epochs')
parser.add_argument('--ncat', type=int, default=1, help='number of feature vectors to concatenate')
args = parser.parse_args()

feature_file = args.features
num_used_features = 20
sequence_length = args.ncat
batch_size = 32

dataset = f32Dataset(feature_file, sequence_length)
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
w1=512
class NeuralNetwork1(nn.Module):
    def __init__(self, input_dim, bottle_dim):
        super().__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(input_dim, w1),
            nn.ReLU(),
            nn.Linear(w1, bottle_dim),
            nn.ReLU(),
            nn.Linear(bottle_dim, w1),
            nn.ReLU(),
            nn.Linear(w1, input_dim)
       )

    def forward(self, x):
        y = self.linear_relu_stack(x)
        return y

# take concatenated vectors
class NeuralNetwork2(nn.Module):
    def __init__(self, input_dim, bottle_dim, seq):
        super().__init__()
        self.input_dim = input_dim
        self.seq = seq
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(input_dim*seq, w1),
            nn.ReLU(),
            nn.Linear(w1, bottle_dim),
            nn.ReLU(),
            nn.Linear(bottle_dim, w1),
            nn.ReLU(),
            nn.Linear(w1, input_dim*seq)
       )

    def forward(self, x):
        x1 = torch.reshape(x,(-1,1,self.input_dim*self.seq))
        #print(x.shape,x1.shape)
        y1 = self.linear_relu_stack(x1)
        y = torch.reshape(y1,(-1,self.seq,self.input_dim))
        #print(y.shape,y1.shape)
        return y

model = NeuralNetwork2(num_used_features, args.bottle_dim, sequence_length).to(device)
print(model)

# criterion to computes the loss between input and target
loss_fn = nn.MSELoss()

# optimizer that will be used to update weights and biases
optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)

for epoch in range(args.epochs):
    sum_loss = 0.0
    for batch, x in enumerate(dataloader):
        x = x.to(device)
        y = model(x)
        loss = loss_fn(x, y)   
        loss.backward() 
        optimizer.step()
        optimizer.zero_grad()
        sum_loss += loss.item()
    print(f'Epochs:{epoch + 1:5d} | ' \
          f'Batches per epoch: {batch + 1:3d} | ' \
          f'Loss: {sum_loss / (batch + 1):.10f}')
