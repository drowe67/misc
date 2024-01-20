"""
Autoencoder attempt 4, to test basic autoencoder performance
"""

import torch
from torch import nn
import numpy as np
import argparse
import torch.nn.functional as F
from matplotlib import pyplot as plt
# vector-quantize-pytorch library
from vector_quantize_pytorch import FSQ, VectorQuantize
# another VQ-VAE
from vqvae import VectorQuantizer

# loading datasets in .f32 files
class f32Dataset(torch.utils.data.Dataset):
    def __init__(self,
                feature_file,
                sequence_length,
                num_features=22,
                num_dB_features=20,
                overlap=True, norm=False, lower_limit_dB=-100, zero_mean=False):

        self.sequence_length = sequence_length
        self.overlap = overlap

        self.features = np.reshape(np.fromfile(feature_file, dtype=np.float32), (-1, num_features))
        self.edB_feature = np.zeros(self.features.shape[0],dtype=np.float32)

        for i in range(self.features.shape[0]):
            e = np.sum(10**(self.features[i,:num_dB_features]/10))
            edB_feature = 10*np.log10(e)
            if edB_feature < lower_limit_dB:
                self.features[i,:num_dB_features] += lower_limit_dB - edB_feature
                edB_feature = lower_limit_dB
            if norm:
                self.features[i,:num_dB_features] -= edB_feature
            self.edB_feature[i] = edB_feature
   
        # features are in dB 20log10(), scale down by 20 to reduce dynamic range but keep log response        
        self.features[:,:num_dB_features] = self.features[:,:num_dB_features]/20

        self.mean= 0
        if zero_mean:
            self.mean = np.mean(self.features[:,:num_dB_features],axis=0)
            self.features[:,:num_dB_features] -= self.mean
        print(self.mean)
        print(np.std(self.features[:,:num_dB_features],axis=0))
        if overlap:
            self.num_sequences = self.features.shape[0] - sequence_length + 1
        else:
            self.num_sequences = self.features.shape[0] // sequence_length

        self.num_sequences1 = self.features.shape[0]

    def __len__(self):
        return self.num_sequences

    def __len1__(self):
        return self.num_sequences1

    # overlapping sequences to make better use of training data
    def __getitem__(self, index):
        if self.overlap:
            features = self.features[index: (index + self.sequence_length), :]
        else:
            features = self.features[index*self.sequence_length: (index+1)*(self.sequence_length), :]      
        return features
    
    def get_edB_feature(self, index):
        return self.edB_feature[index]

    def get_mean(self):
        return self.mean

parser = argparse.ArgumentParser()
parser.add_argument('features', type=str, help='path to feature file in .f32 format')
parser.add_argument('--lr', type=float, default=5E-2, help='learning rate')
parser.add_argument('--bottle_dim', type=int, default=10, help='bottleneck dim')
parser.add_argument('--epochs', type=int, default=10, help='number of training epochs')
parser.add_argument('--ncat', type=int, default=1, help='number of feature vectors to concatenate')
parser.add_argument('--save_model', type=str, default="", help='filename of model to save')
parser.add_argument('--inference', type=str, default="", help='Inference only with filename of saved model')
parser.add_argument('--out_file', type=str, default="", help='path to output file b_hat[22] in .f32 format')
parser.add_argument('--write_latent', type=str, default="", help='path to output file of latent vectors l[bottle_dim] in .f32 format')
parser.add_argument('--read_latent', type=str, default="", help='path to output file of latent vectors l[bottle_dim] in .f32 format')
parser.add_argument('--noplot', action='store_true', help='disable plots after training')
parser.add_argument('--frame', type=int, default=165, help='frame # to start viewing')
parser.add_argument('--nn', type=int, default=2, help='Neural Network to use')
parser.add_argument('--norm', action='store_true', help='normalise energy')
parser.add_argument('--nvq', type=int, default=1, help='number of vector quantisers')
parser.add_argument('--wloss', action='store_true', help='use weighted linear loss function')
parser.add_argument('--noise_var', type=float, default=0.0, help='inject gaussian noise at bottleneck')
parser.add_argument('--lower_limit_dB', type=float, default=10.0, help='lower limit in energy per feature vector')
parser.add_argument('--zero_mean', action='store_true', help='remove mean from training data')
parser.add_argument('--loss_file', type=str, default="", help='file with epoch\tloss on each line')
parser.add_argument('--opt', type=str, default="SGD", help='SGD/Adam')
args = parser.parse_args()

feature_file = args.features
num_features = 22
num_used_features = 20
sequence_length = args.ncat
batch_size = 32
gamma = 0.5
inject_l_hat = args.write_latent;

dataset = f32Dataset(feature_file, sequence_length, norm=args.norm,lower_limit_dB=args.lower_limit_dB, zero_mean=args.zero_mean)
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

# consistent good performer
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
        return y,torch.zeros((1))

# quite good, but stuck at 0.2 after 100 epochs
class NeuralNetwork2(nn.Module):
    def __init__(self, input_dim, bottle_dim):
        super().__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, w1),
            nn.ReLU(),
            nn.Linear(w1, bottle_dim),
            nn.ReLU(),
            nn.Linear(bottle_dim, w1),
            nn.ReLU(),
            nn.Linear(w1, 64),
            nn.ReLU(),
            nn.Linear(64, input_dim)
       )

    def forward(self, x):
        y = self.linear_relu_stack(x)
        return y,torch.zeros((1))

# poor perf, around 1dB after 50 epochs
class NeuralNetwork3(nn.Module):
    def __init__(self, input_dim, bottle_dim):
        super().__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, bottle_dim),
            nn.ReLU(),
            nn.Linear(bottle_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, input_dim)
       )

    def forward(self, x):
        y = self.linear_relu_stack(x)
        return y,torch.zeros((1))


class NeuralNetwork4(nn.Module):
    def __init__(self, input_dim, bottle_dim):
        super().__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(input_dim, w1),
            nn.ReLU(),
            nn.Linear(w1, w1/2),
            nn.ReLU(),
            nn.Linear(w1, bottle_dim),
            nn.ReLU(),
            nn.Linear(bottle_dim, w1),
            nn.ReLU(),
            nn.Linear(w1, w1/2),
            nn.ReLU(),
            nn.Linear(w1, input_dim)
       )

    def forward(self, x):
        y = self.linear_relu_stack(x)
        return y,torch.zeros((1))
match args.nn:
    case 1:
        model = NeuralNetwork1(num_used_features, args.bottle_dim).to(device)
    case 2:
        model = NeuralNetwork2(num_used_features, args.bottle_dim).to(device)
    case 3:
        model = NeuralNetwork3(num_used_features, args.bottle_dim).to(device)
    case 4:
        model = NeuralNetwork4(num_used_features, args.bottle_dim).to(device)
    case _:
        print("unknown network!")
        quit()

if len(args.inference) == 0:
    print(model)
    num_weights = sum(p.numel() for p in model.parameters())
    print(f"weights: {num_weights} float32 memory: {num_weights*4}")

# PyTorch custom loss function that operates in the weighted linear domain
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
    return loss

if len(args.inference) == 0:
    # criterion to computes the loss between input and target
    if args.wloss:
        loss_fn =  my_loss
        print("training with weighted linear loss")
    else:
        loss_fn =  nn.MSELoss()
        print("training with MSE loss")

    # optimizer that will be used to update weights and biases
    if args.opt == "SGD":
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)
        print("SGD")
    if args.opt == "Adam":
        optimizer = torch.optim.Adam(model.parameters(), weight_decay=args.lr)
        print("Adam")
   
    loss_epoch=np.zeros((args.epochs))

    for epoch in range(args.epochs):
        sum_loss = 0.0
        for batch, x in enumerate(dataloader):
            # strip off Wo and v features
            x = x[:,:,:num_used_features] 
            x = x.to(device)
            y,l = model(x)
            loss = loss_fn(x, y)
               
            loss.backward() 
            optimizer.step()
            optimizer.zero_grad()
            sum_loss += loss.item()
   
        # Convert back to dB^2
        sum_loss_dB2 = sum_loss * 400
        print(f'Epochs:{epoch + 1:5d} | ' \
            f'Batches per epoch: {batch + 1:3d} | ' \
            f'Loss: {sum_loss_dB2 / (batch + 1):5.2f}')
        loss_epoch[epoch] = sum_loss_dB2 / (batch + 1)

    if len(args.loss_file):
        np.savetxt(args.loss_file, loss_epoch)

    if len(args.save_model):
        print(f"Saving model to: {args.save_model}")
        torch.save(model.state_dict(), args.save_model)

# load a model file, run test data through it 
if len(args.inference):
    print(f"Loading model from: {args.inference}")
    model.load_state_dict(torch.load(args.inference))
    model.eval()
    dataset_inference = f32Dataset(feature_file, sequence_length, norm=args.norm, overlap=False,lower_limit_dB=args.lower_limit_dB,zero_mean=args.zero_mean)
    len_out = dataset_inference.__len__()
    len_out1 = dataset_inference.__len1__()
    #print(len_out1, len_out)
    b_hat_np = np.ones((len_out1,num_features),dtype=np.float32)
    # latent vectors out of model
    l_np = np.ones((len_out1,args.bottle_dim),dtype=np.float32)

    # optionally read latent vectors from file for injection into network
    if inject_l_hat:
        l_hat = np.fromfile(args.read_latent, dtype=np.float32)
        #print(l_hat.shape)
        l_hat = l_hat.reshape((-1),args.bottle_dim)
        #print(l_hat.shape)
  
    with torch.no_grad():
        sum_sd = 0
        for i in range(len_out):
            b = dataset_inference.__getitem__(i)
            b1 = b[:,:num_used_features].reshape((1,sequence_length,num_used_features))
            if inject_l_hat:
                b_hat,l = model(torch.from_numpy(b1).to(device), torch.from_numpy(l_hat[i,:]).to(device))
            else:
                b_hat,l = model(torch.from_numpy(b1).to(device))
               
            b_hat_cpu = b_hat[0,:].cpu().numpy()
            l_cpu = l.cpu().numpy()

            sum_sd = sum_sd + 400*np.mean((b1-b_hat_cpu)**2)
            st = i*sequence_length
            en = (i+1)*sequence_length
            b_hat_np[st:en,:20] = 20*b_hat[0,]
            # put Wo and v back into features
            #print(st,en)
            b_hat_np[st:en,num_used_features:num_features] = b[:,num_used_features:num_features]
            l_np[i,:] = l_cpu

    print(f"Eq:{sum_sd/len_out:5.2f}")

    if len(args.out_file):
        #print(b_hat_np.shape)
        b_hat_np = b_hat_np.reshape((-1))
        #print(b_hat_np.shape)
        b_hat_np.astype('float32').tofile(args.out_file)
    if len(args.write_latent):
        #print(l_np.shape)
        l_np = l_np.reshape((-1))
        #print(l_np.shape)
        l_np.astype('float32').tofile(args.write_latent)

# interactive frame-frame visualisation of running model on test data
if args.noplot == False:

    # we may have already loaded test data if in inference mode
    if len(args.inference) == 0:
        model.eval()
        dataset_inference = f32Dataset(feature_file, sequence_length, norm=args.norm, overlap=False, lower_limit_dB=args.lower_limit_dB, zero_mean=args.zero_mean)
        len_out = dataset_inference.__len__()

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
    if sequence_length == 1:
        ax = np.array([ax])
    print(ax.shape)
    fig.canvas.mpl_connect('key_press_event', on_press)

    with torch.no_grad():
        f = args.frame // sequence_length
        loop = True
        amean = dataset_inference.get_mean()
        while loop:
            b = dataset_inference.__getitem__(f)
            b = b[:,:num_used_features].reshape((1,sequence_length,num_used_features))
            if inject_l_hat:
                b_hat,l = model(torch.from_numpy(b1).to(device), torch.from_numpy(l_hat[f,:]).to(device))
            else:
                b_hat,l = model(torch.from_numpy(b).to(device))
            b_plot = 20*(b[0,] + amean)
            b_hat_plot = 20*(b_hat[0,].cpu().numpy() + amean)
            for j in range(sequence_length):
                ax[j].cla()
                if args.norm:
                    edB = dataset_inference.get_edB_feature(f*sequence_length+j)
                else:
                    edB = 0
                ax[j].plot(b_f_kHz,edB+b_plot[j,0:20])
                t = f"f: {f*sequence_length+j}"
                ax[j].set_title(t)
                #print(dataset_inference.get_edB_feature(f+j))
                ax[j].plot(b_f_kHz,edB+b_hat_plot[j,0:20],'r')
                ax[j].axis([0, 4, -20, 70])
 
            plt.show(block=False)
            plt.pause(0.01)
            button = plt.waitforbuttonpress(0)
            if akey == 'b':
                if f:
                    f -= 1
            if akey == 'n' or button == False:
                if f < len_out:
                    f += 1
            if akey == 'j':
                f = test_frames[test_frames_ind]
                test_frames_ind += 1
                test_frames_ind = np.mod(test_frames_ind,4)
            if akey == 'q':
                loop = False
            akey = ''
