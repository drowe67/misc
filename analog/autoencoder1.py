"""
Autoencoder attempt 1, to gain experience in concatenating multiple frames of features
"""

import torch
from torch import nn
import numpy as np
import argparse
import torch.nn.functional as F
from matplotlib import pyplot as plt

# loading datasets in .f32 files
class f32Dataset(torch.utils.data.Dataset):
    def __init__(self,
                feature_file,
                sequence_length,
                num_used_features=20,
                num_features=22,
                overlap=True):

        self.sequence_length = sequence_length
        self.overlap = overlap

        # features are in dB 20log10(), remove 20 to reduce dynamic range but keep log response
        self.features = np.reshape(np.fromfile(feature_file, dtype=np.float32), (-1, num_features))/20
        self.features = self.features[:, :num_used_features]
        if overlap:
            self.num_sequences = self.features.shape[0] - sequence_length + 1
        else:
            self.num_sequences = self.features.shape[0] // sequence_length
           
    def __len__(self):
        return self.num_sequences

    # overlapping sequences to make better use of training data
    def __getitem__(self, index):
        if self.overlap:
            features = self.features[index: (index + self.sequence_length), :]
        else:
            features = self.features[index*self.sequence_length: (index+1)*(self.sequence_length), :]      
        return features
    
parser = argparse.ArgumentParser()
parser.add_argument('features', type=str, help='path to feature file in .f32 format')
parser.add_argument('--lr', type=float, default=5E-2, help='learning rate')
parser.add_argument('--bottle_dim', type=int, default=10, help='bottleneck dim')
parser.add_argument('--epochs', type=int, default=10, help='number of training epochs')
parser.add_argument('--ncat', type=int, default=1, help='number of feature vectors to concatenate')
parser.add_argument('--save_model', type=str, default="", help='filename of model to save')
parser.add_argument('--inference', type=str, default="", help='Inference only with filename of saved model')
parser.add_argument('--out_file', type=str, default="", help='path to output file [y[79]] in .f32 format')
parser.add_argument('--noplot', action='store_true', help='disable plots after training')
parser.add_argument('--frame', type=int, default=165, help='frame # to start viewing')
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

# concatenated vectors
class NeuralNetwork2(nn.Module):
    def __init__(self, input_dim, bottle_dim, seq):
        super().__init__()
        self.input_dim = input_dim
        self.seq = seq
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(input_dim*seq, w1),
            nn.ReLU(),
            nn.Linear(w1, bottle_dim),
            nn.Tanh(),
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
        #print(x)
        return y

# conv1d
class NeuralNetwork3(nn.Module):
    def __init__(self, input_dim, bottle_dim, seq, nf=128):
        super().__init__()
        self.input_dim = input_dim
        self.seq = seq
        self.nf = nf
        self.c1 = nn.Conv1d(input_dim, nf, 3, padding='same')
        self.c2 = nn.Conv1d(nf, bottle_dim, 3, padding='same')
        self.c3 = nn.Conv1d(bottle_dim, nf, 3, padding='same')
        self.c4 = nn.Conv1d(nf, input_dim, 3, padding='same')
    
    def forward(self, x):

        # get reshape to (batch, features, timesteps) or in Torch terms (batch, channels, length))
        #print(x.shape)
        x = x.permute((0,2,1))
        #print(x.shape)

        # encoder
        x = F.relu(self.c1(x))
        x = F.max_pool1d(x,2)
        #print(x.shape)
        x = F.relu(self.c2(x))
        x = F.max_pool1d(x,2)
        #print(x.shape)

        # decoder
        x = F.interpolate(x,2)
        x = F.relu(self.c3(x))
        #print(x.shape)
        x = F.interpolate(x,4)
        #print(x.shape)
        x = self.c4(x)
        #print(x.shape)
        x = x.permute((0,2,1))
        #print(x.shape)
        #quit()

        return x

# concatenated vectors, add some noise to the bottleneck
class NeuralNetwork4(nn.Module):
    def __init__(self, input_dim, bottle_dim, seq):
        super().__init__()
        self.input_dim = input_dim
        self.seq = seq
        self.bottle_dim = bottle_dim
        self.l1 = nn.Linear(input_dim*seq, w1)
        self.l2 = nn.Linear(w1, bottle_dim)
        self.l3 = nn.Linear(bottle_dim,w1)
        self.l4 = nn.Linear(w1, input_dim*seq)
 
    def forward(self, x):
        x = torch.reshape(x,(-1,1,self.input_dim*self.seq))
        #print(x.shape,x1.shape)
        x = F.relu(self.l1(x))
        x = F.tanh(self.l2(x))
        x = x + (0.01**0.5)*torch.randn(1,self.bottle_dim)
        x = F.relu(self.l3(x))
        x = self.l4(x)
        x = torch.reshape(x,(-1,self.seq,self.input_dim))
        #print(y.shape,y1.shape)
        return x

model = NeuralNetwork2(num_used_features, args.bottle_dim, sequence_length).to(device)
print(model)

if len(args.inference) == 0:
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
    
        sum_loss_dB2 = sum_loss * 400
        print(f'Epochs:{epoch + 1:5d} | ' \
            f'Batches per epoch: {batch + 1:3d} | ' \
            f'Loss: {sum_loss_dB2 / (batch + 1):5.2f} dB^2')

    if len(args.save_model):
        print(f"Saving model to: {args.save_model}")
        torch.save(model.state_dict(), args.save_model)

# load a model file, run test data through it 
if len(args.inference):
    print(f"Loading model from: {args.inference}")
    model.load_state_dict(torch.load(args.inference))
    model.eval()
    dataset_inference = f32Dataset(feature_file, sequence_length, overlap=False)
    len_out = dataset_inference.__len__()
    print(len_out)
    b_hat_np = np.zeros([len_out,num_used_features*sequence_length],dtype=np.float32)
    with torch.no_grad():
        sum_Eq = 0
        for i in range(len_out):
            b = dataset_inference.__getitem__(i)
            b_hat = model(torch.from_numpy(b).to(device))
            b_hat_cpu = b_hat[0,:].cpu().numpy()
            
            sum_Eq = sum_Eq + 400*np.mean((b-b_hat_cpu)**2)
            b_hat_np[i,] = 20*b_hat[0,].reshape((num_used_features*sequence_length))
    print(f"Eq:{sum_Eq/len_out:5.2f}")

    if len(args.out_file):
        print(b_hat_np.shape)
        b_hat_np = b_hat_np.reshape((-1))
        print(b_hat_np.shape)
        b_hat_np.astype('float32').tofile(args.out_file)

# interactive frame-frame visualisation of running model on test data
if args.noplot == False:

    # we may have already loaded test data if in inference mode
    if len(args.inference) == 0:
        model.eval()
        # num_test == 0 switches off energy and V filtering, so we get all frames in test data.
        dataset_inference = f32Dataset(feature_file, sequence_length, overlap=False)

    print("[click or n]-next [b]-back [j]-jump [w]-weighting [q]-quit")

    b_f_kHz = np.array([0.1998, 0.2782, 0.3635, 0.4561, 0.5569, 0.6664, 0.7855, 0.9149, 1.0556, 1.2086, 1.3749, 1.5557,
    1.7523, 1.9659, 2.1982, 2.4508, 2.7253, 3.0238, 3.3483, 3.7011])
    Fs = 8000

    # some interesting frames male1,male1,male2,female, use 'j' to jump to them
    test_frames = np.array([61, 165, 4190, 5500])
    test_frames_ind = 0

    # called when we press a key on the plot
    akey = ''
    def on_press(event):
        global akey
        akey = event.key

    fig, ax = plt.subplots(sequence_length, 1)
    fig.canvas.mpl_connect('key_press_event', on_press)
    ax_b = ax[0].twinx()

    with torch.no_grad():
        f = args.frame // sequence_length
        loop = True
        while loop:
            b = dataset_inference.__getitem__(f)
            b_hat = model(torch.from_numpy(b).to(device))
            b_plot = 20*b
            b_hat_plot = 20*b_hat[0,].cpu().numpy()
            for j in range(sequence_length):
                ax[j].cla()
                ax[j].plot(b_f_kHz,b_plot[j,0:20])
                t = f"f: {f+j}"
                ax[j].set_title(t)
                ax[j].plot(b_f_kHz,b_hat_plot[j,0:20],'r')
                ax[j].axis([0, 4, 0, 70])
 
            plt.show(block=False)
            plt.pause(0.01)
            button = plt.waitforbuttonpress(0)
            if akey == 'b':
                f -= 1
            if akey == 'n' or button == False:
                f += 1
            if akey == 'j':
                f = test_frames[test_frames_ind]
                test_frames_ind += 1
                test_frames_ind = np.mod(test_frames_ind,4)
            if akey == 'q':
                loop = False
            akey = ''
