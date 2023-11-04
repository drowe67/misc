"""
Test VQVAE by getting it to train on a QPSK constellation
"""

import torch
from torch import nn
import numpy as np
import argparse
from vqvae import VectorQuantizer
from matplotlib import pyplot as plt

class QPSKDataset(torch.utils.data.Dataset):
    def __init__(self,
                embedding_dim,
                num_samples):

        self.num_samples = num_samples

        # a QPSK constellation with noise
        bits = np.random.randint(2,size=num_samples*embedding_dim).reshape(num_samples, embedding_dim)
        self.x_train = (2*bits-1 + 0.1*np.random.randn(num_samples, embedding_dim)).astype(np.float32)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):
        x = self.x_train[index, :]
        return x

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
    def __init__(self, embedding_dim, num_embeddings):
        super().__init__()
        self.vq = VectorQuantizer(embedding_dim, num_embeddings)
        self.lin = nn.Linear(embedding_dim, embedding_dim)
    def forward(self, x):
        z = self.lin(x)
        (z_quantized, dictionary_loss, commitment_loss, encoding_indices) = self.vq(z)
        return {
            "dictionary_loss": dictionary_loss,
            "commitment_loss": commitment_loss,
            "x_recon": z_quantized,
        }

embedding_dim = 2
num_embedding = 4
num_samples = 1024
batch_size = 4

dataset = QPSKDataset(embedding_dim, num_samples)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)

for f in dataloader:
    print(f"Shape of features: {f.shape}")
    break

model = NeuralNetwork(embedding_dim, num_embedding).to(device)
print(model)

loss_fn = nn.MSELoss()
train_params = [params for params in model.parameters()]
print(train_params)
optimizer = torch.optim.SGD(train_params, lr=0.01)

epochs = 3
for epoch in range(epochs):
    running_loss = 0.0
    for batch, x in enumerate(dataloader):
        x = x.to(device)
        y = model(x)
        #print(y)
        loss = loss_fn(x, y["x_recon"])   
        loss.backward() 
        optimizer.step()
        optimizer.zero_grad()
        
    running_loss += loss.item()
    print(f'Epochs:{epoch + 1:5d} | ' \
          f'Batches per epoch: {batch + 1:3d} | ' \
          f'Loss: {running_loss / (batch + 1):.10f}')

for param in model.parameters():
  print(param.data)

x_train = np.empty((0,2))
x_pred = np.empty((0,2))
with torch.no_grad():
    for x in dataloader:
        x = x.to(device)
        y = model(x)
        x_train = np.append(x_train, x.cpu(), axis=0)
        x_pred = np.append(x_pred, y["x_recon"].cpu(), axis=0)
print(x_train.shape,x_pred.shape)

plt.scatter(x_train[:,0],x_train[:,1],label='train')
plt.scatter(x_pred[:,0],x_pred[:,1],label='pred')
plt.legend()
plt.show()
