"""
Autoencoder attempt 1, to gain experience in concatenating multiple frames of features
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
                overlap=True, norm=False):

        self.sequence_length = sequence_length
        self.overlap = overlap

        self.features = np.reshape(np.fromfile(feature_file, dtype=np.float32), (-1, num_features))
        self.edB_feature = np.zeros(self.features.shape[0],dtype=np.float32)

        if norm:
            for i in range(self.features.shape[0]):
                e = np.sum(10**(self.features[i,:num_dB_features]/10))
                edB_feature = 10*np.log10(e)
                self.features[i,:num_dB_features] -= edB_feature
                self.edB_feature[i] = edB_feature
           
        # features are in dB 20log10(), scale down by 20 to reduce dynamic range but keep log response        
        self.features[:,:num_dB_features]= self.features[:,:num_dB_features]/20
        #self.amean = np.mean(self.features,axis=0)
        self.amean = np.zeros(num_features)
        print(np.mean(self.features,axis=0))
        print(np.std(self.features,axis=0))
        print(self.amean.shape, self.features.shape)
        #self.features -= self.amean
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

    def get_amean(self):
        return self.amean

parser = argparse.ArgumentParser()
parser.add_argument('features', type=str, help='path to feature file in .f32 format')
parser.add_argument('--lr', type=float, default=5E-2, help='learning rate')
parser.add_argument('--bottle_dim', type=int, default=10, help='bottleneck dim')
parser.add_argument('--epochs', type=int, default=10, help='number of training epochs')
parser.add_argument('--ncat', type=int, default=1, help='number of feature vectors to concatenate')
parser.add_argument('--save_model', type=str, default="", help='filename of model to save')
parser.add_argument('--inference', type=str, default="", help='Inference only with filename of saved model')
parser.add_argument('--out_file', type=str, default="", help='path to output file [b_hat[22]] in .f32 format')
parser.add_argument('--noplot', action='store_true', help='disable plots after training')
parser.add_argument('--frame', type=int, default=165, help='frame # to start viewing')
parser.add_argument('--nn', type=int, default=2, help='Neural Network to use')
parser.add_argument('--norm', action='store_true', help='normalise energy')
parser.add_argument('--nvq', type=int, default=1, help='number of vector quantisers')
args = parser.parse_args()

feature_file = args.features
num_features = 22
num_used_features = 20
sequence_length = args.ncat
batch_size = 32
gamma = 0.5
num_embeddings = 512

dataset = f32Dataset(feature_file, sequence_length, norm=args.norm)
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

# Toy FSQ example, one VQ step per sequence, we would expect efficiency to increase
# with sequence as receptive field is larger
w5=512
class NeuralNetwork5(nn.Module):
    def __init__(self, input_dim, bottle_dim, seq, nvq, nlevels=8):
        super().__init__()
        self.input_dim = input_dim
        self.seq = seq
        self.bottle_dim = bottle_dim
        self.nvq = nvq

        self.l1 = nn.Linear(input_dim*seq, w5)
        #self.l1a = nn.Linear(w5,w5)
        #self.l1b = nn.Linear(w5,w5)
        #self.l1c = nn.Linear(w5,w5)
        self.l2 = nn.Linear(w5, bottle_dim*nvq)
        levels = nlevels*np.ones(bottle_dim)
        print(levels.tolist(), nlevels**bottle_dim, np.log2(nlevels**bottle_dim))
        self.quantizer = FSQ(levels.tolist())
        self.l3 = nn.Linear(bottle_dim*nvq,w5)
        #self.l3a = nn.Linear(w5,w5)
        #self.l3b = nn.Linear(w5,w5)
        #self.l3c = nn.Linear(w5,w5)
        self.l4 = nn.Linear(w5, input_dim*seq)
 
    def forward(self, x):
        x = torch.reshape(x,(-1,1,self.input_dim*self.seq))
        #print(x.shape)
        x = F.relu(self.l1(x))
        #x = F.relu(self.l1a(x))
        #x = F.relu(self.l1b(x))
        #x = F.relu(self.l1c(x))
        x = F.tanh(self.l2(x))
        x = x.reshape(x.shape[0],self.nvq,self.bottle_dim)
        #print(x.shape)
        xhat, indices = self.quantizer(x)
        #print(xhat.shape)
        #quit()
        x = F.relu(self.l3(xhat.reshape((xhat.shape[0],self.nvq*self.bottle_dim))))
        #x = F.relu(self.l3a(x))
        #x = F.relu(self.l3b(x))
        #x = F.relu(self.l3c(x))
        x = self.l4(x)
        x = torch.reshape(x,(-1,self.seq,self.input_dim))
        #print(y.shape,y1.shape)
        return x

# Toy VQVAE example, one VQ step per sequence, TODO: need more complex loss function
w6=512
class NeuralNetwork6(nn.Module):
    def __init__(self, input_dim, bottle_dim, seq, nvq):
        super().__init__()
        self.input_dim = input_dim
        self.seq = seq
        self.bottle_dim = bottle_dim
        self.nvq = nvq

        self.l1 = nn.Linear(input_dim*seq, w5)
        #self.l1a = nn.Linear(w5,w5)
        self.l2 = nn.Linear(w5, bottle_dim*nvq)
        self.vq = VectorQuantize(
            dim = bottle_dim,
            codebook_size = 512,     # codebook size
            decay = 0.8,             # the exponential moving average decay, lower means the dictionary will change faster
            commitment_weight = 1.   # the weight on the commitment loss
        )
        self.l3 = nn.Linear(bottle_dim*nvq,w5)
        #self.l3a = nn.Linear(w5,w5)
        self.l4 = nn.Linear(w5, input_dim*seq)
 
    def forward(self, x):
        x = torch.reshape(x,(-1,1,self.input_dim*self.seq))
        #print(x.shape)
        x = F.relu(self.l1(x))
        #x = F.relu(self.l1a(x))
        x = F.relu(self.l2(x))
        x = x.reshape(x.shape[0],self.nvq,self.bottle_dim)
        #print(x.shape)
        xhat, indices, commit_loss = self.vq(x)
        #print(xhat.shape)
        #quit()
        x = F.relu(self.l3(xhat.reshape((xhat.shape[0],self.nvq*self.bottle_dim))))
        #x = F.relu(self.l3a(x))
        x = self.l4(x)
        x = torch.reshape(x,(-1,self.seq,self.input_dim))
        #print(y.shape,y1.shape)
        return x, commit_loss

# Another VQVAE example, lets just try a straight VQ
class NeuralNetwork7(nn.Module):
    def __init__(self, input_dim, bottle_dim, seq, nvq):
        super().__init__()
        self.input_dim = input_dim
        self.seq = seq
        self.bottle_dim = bottle_dim
        self.nvq = nvq

        # Torch chokes if we don't have a trainable layer
        self.lin = nn.Linear(input_dim, input_dim)       
        self.vq1 = VectorQuantizer(input_dim, num_embeddings,decay=0.99)
        self.vq2 = VectorQuantizer(input_dim, num_embeddings,decay=0.99)
        self.vq3 = VectorQuantizer(input_dim, num_embeddings,decay=0.99)
        self.vq = nn.ModuleList([VectorQuantizer(input_dim, num_embeddings,decay=0.99) for i in range(nvq)])

    def forward(self, x):
        x = x.reshape(x.shape[0],self.input_dim)
        
        y = torch.zeros(x.shape)
        commitment_loss_stage = torch.zeros(self.nvq)
        encoding_indices_stage = torch.zeros(self.nvq,x.shape[0],dtype=torch.int64)
        for i in range(self.nvq):
            (x_hat, dictionary_loss, commitment_loss, encoding_indices) = self.vq[i](x)
            x = x - x_hat            # VQ residual for next stage (note we can't use x -= x_hat)
            y = y + x_hat            # VQ output is sum of all VQs
            #print(encoding_indices.dtype)
            #quit()
            commitment_loss_stage[i] = commitment_loss
            encoding_indices_stage[i,:] = encoding_indices[:,0]
        #x = y
        
        """
        (x_hat, dictionary_loss, commitment_loss, encoding_indices) = self.vq[0](x)
        #x_res = x - x_hat
        x = x - x_hat
        #print(y.shape,x.shape,x_hat.shape)
        #quit()
        y += x_hat
        (x_hat, dictionary_loss, commitment_loss2, encoding_indices2) = self.vq[1](x)
        x = x - x_hat
        y += x_hat
        (x_hat, dictionary_loss, commitment_loss3, encoding_indices3) = self.vq[2](x)
        #x = x_hat + x_res_hat + x_res2_hat
        y += x_hat
        """
        #x = y
        y = y.reshape(y.shape[0],1,self.input_dim)
        y = self.lin(y)

        # don't think we need to return commitment loss if no trainable input layer
        return {
            "y": y,
            "encoding_indices": encoding_indices_stage[0,:],
            "commitment_loss": commitment_loss_stage[0],
        }


match args.nn:
    case 1:
        model = NeuralNetwork1(num_used_features, args.bottle_dim, sequence_length).to(device)
    case 2:
        model = NeuralNetwork2(num_used_features, args.bottle_dim, sequence_length).to(device)
    case 3:
        model = NeuralNetwork3(num_used_features, args.bottle_dim, sequence_length).to(device)
    case 4:
        model = NeuralNetwork4(num_used_features, args.bottle_dim, sequence_length).to(device)
    case 5:
        model = NeuralNetwork5(num_used_features, args.bottle_dim, sequence_length, args.nvq).to(device)
    case 6:
        model = NeuralNetwork6(num_used_features, args.bottle_dim, sequence_length, args.nvq).to(device)
    case 7:
        model = NeuralNetwork7(num_used_features, args.bottle_dim, sequence_length, args.nvq).to(device)
    case _:
        print("unknown network!")
        quit()

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
    #loss_fn =  my_loss
    loss_fn =  nn.MSELoss()

    # optimizer that will be used to update weights and biases
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)

    for epoch in range(args.epochs):
        sum_loss = 0.0
        sum_loss_total = 0.0
        indices_map = np.zeros(num_embeddings)
        for batch, x in enumerate(dataloader):
            # strip off Wo and v features
            x = x[:,:,:num_used_features] 
            x = x.to(device)
            y = model(x)
            indices = y["encoding_indices"].cpu().numpy()
            indices_map[indices] = 1
            loss = loss_fn(x, y["y"]) + 0.25*y["commitment_loss"]
            loss.backward() 
            optimizer.step()
            optimizer.zero_grad()
            #print(loss, loss.item())
            sum_loss_total += loss.item()
            sum_loss += loss_fn(x, y["y"])
    
        sum_loss_total_dB2 = sum_loss_total * 400
        sum_loss_dB2 = sum_loss * 400
        vq_util = np.mean(indices_map)
        print(f'Epochs:{epoch + 1:5d} | ' \
            f'Batches per epoch: {batch + 1:3d} | ' \
            f'Loss: {sum_loss_dB2 / (batch + 1):5.2f} ' \
            f'Loss Total: {sum_loss_total_dB2 / (batch + 1):5.2f} ' \
             f'VQ util: {vq_util:3.2f}')

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
    len_out1 = dataset_inference.__len1__()
    print(len_out1, len_out)
    b_hat_np = np.ones((len_out1,num_features),dtype=np.float32)

    with torch.no_grad():
        sum_sd = 0
        for i in range(len_out):
            b = dataset_inference.__getitem__(i)
            b1 = b[:,:num_used_features].reshape((1,sequence_length,num_used_features))
            b_hat = model(torch.from_numpy(b1).to(device))
            b_hat_cpu = b_hat[0,:].cpu().numpy()
            
            sum_sd = sum_sd + 400*np.mean((b1-b_hat_cpu)**2)
            st = i*sequence_length
            en = (i+1)*sequence_length
            b_hat_np[st:en,:20] = 20*b_hat[0,]
            # put Wo and v back into features
            #print(st,en)
            b_hat_np[st:en,num_used_features:num_features] = b[:,num_used_features:num_features]
    print(f"Eq:{sum_sd/len_out:5.2f}")

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
        dataset_inference = f32Dataset(feature_file, sequence_length, norm=args.norm, overlap=False)
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
        amean = 20*dataset_inference.get_amean()[:num_used_features]
        while loop:
            b = dataset_inference.__getitem__(f)
            b = b[:,:20].reshape((1,sequence_length,num_used_features))
            b_hat = model(torch.from_numpy(b).to(device))
            b_plot = 20*b[0,]
            b_hat_plot = 20*b_hat["y"][0,].cpu().numpy()
            for j in range(sequence_length):
                ax[j].cla()
                edB = dataset_inference.get_edB_feature(f*sequence_length+j) + amean
                ax[j].plot(b_f_kHz,edB+b_plot[j,0:20])
                t = f"f: {f*sequence_length+j}"
                ax[j].set_title(t)
                #print(dataset_inference.get_edB_feature(f+j))
                ax[j].plot(b_f_kHz,edB+b_hat_plot[j,0:20],'r')
                ax[j].axis([0, 4, 0, 70])
 
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
