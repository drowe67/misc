"""
autoencoder2.py - ML quantisation experiment 

Based on misc/manifold/manifold.py

y -> bottleneck -> y_hat
"""

import torch
from torch import nn
import numpy as np
import argparse,sys
from matplotlib import pyplot as plt
import torch.nn.functional as F

class f32Dataset(torch.utils.data.Dataset):
    def __init__(self,
                feature_file,
                target_file,
                sequence_length, 
                features_dim,
                target_dim,
                num_test = 0,
                thresh_dB=10):

        self.sequence_length = sequence_length

        features_in = np.reshape(np.fromfile(feature_file, dtype=np.float32,offset=4*num_test*features_dim), (-1, features_dim))
        targets_in = np.reshape(np.fromfile(target_file, dtype=np.float32,offset=4*num_test*target_dim), (-1, target_dim))
        num_sequences_in = features_in.shape[0]

        assert(features_in.shape[0] == targets_in.shape[0])
        
        self.features = np.zeros(features_in.shape,dtype=np.float32)
        self.targets = np.zeros(targets_in.shape,dtype=np.float32)
        self.edB_target = np.zeros(targets_in.shape[0],dtype=np.float32)

        # normalise energy in each vector to 1.0: (a) we are interested NN matching shape, not gain (b)
        # keeps each loss on a similar scale to help gradients (c) a gain difference has a large
        # impact on loss

        j = 0
        for i in range(num_sequences_in):
            e = np.sum(10**(features_in[i,:20]/10))
            edB_feature = 10*np.log10(e)
            # when training, just use voiced vectors above edB_thresh, to avoid training on noise
            
            if (edB_feature > thresh_dB and features_in[i,21]) or num_test == 0:
                self.features[j,] = features_in[i,]
                #print(self.features[j,])
                self.features[j,:20] -= edB_feature
                self.targets[j,] = targets_in[i,]
                e = np.sum(10**(targets_in[i,]/10))
                self.edB_target[j] = 10*np.log10(e)
                self.targets[j,] -= self.edB_target[j]
                #print(self.targets[j,])
 
                # b and y vectors are in x_dB = 20*log10(x), scale down to log10(x).  We don't need to scale
                # Wo and voicing (last two floats in feature vector)
                self.features[j,:20] = self.features[j,:20]/20
                self.targets[j,] = self.targets[j,]/20

                # log10(Wo) probably more useful - but have found results don't change when this feature is unused
                self.features[j,20] = np.log10(self.features[j,20])

                j += 1
        self.num_sequences = j
        if num_test:
            print(f"Training: First {num_test} reserved for testing, {self.num_sequences}/{num_sequences_in} voiced vectors with energy > {thresh_dB} dB used")
        else:
            print(f"Testing: {self.num_sequences} loaded")

    def __len__(self):
        return self.num_sequences

    def __getitem__(self, index):
        features = self.features[index * self.sequence_length: (index + 1) * self.sequence_length, :]
        targets = self.targets[index * self.sequence_length: (index + 1) * self.sequence_length, :]
        return features, targets

    def get_edB_target(self, index):
        return self.edB_target[index]
   
parser = argparse.ArgumentParser()
parser.add_argument('features', type=str, help='path to feature file [b[22] Wo v] .f32 format')
parser.add_argument('target', type=str, help='path to target file [y[79]] in .f32 format')
parser.add_argument('--frame', type=int, default=165, help='frame # to start viewing')
parser.add_argument('--noplot', action='store_true', help='disable plots after training')
parser.add_argument('--num_test', type=int, default=60*100, help='number of vectors reserved for testing')
parser.add_argument('--epochs', type=int, default=10, help='number of training epochs')
parser.add_argument('--thresh', type=float, default=10, help='energy threshold for training data in dB')
parser.add_argument('--lr', type=float, default=5E-2, help='learning rate')
parser.add_argument('--save_model', type=str, default="", help='filename of model to save')
parser.add_argument('--inference', type=str, default="", help='Inference only with filename of saved model')
parser.add_argument('--out_file', type=str, default="", help='path to output file [y[79]] in .f32 format')
parser.add_argument('--bottle_dim', type=int, default=10, help='bottleneck dim')
parser.add_argument('--write_latent', type=str, default="", help='path to output file of latent vectors l[bottle_dim] in .f32 format')
parser.add_argument('--nn', type=int, default=2, help='Neural Network to use')
parser.add_argument('--gamma', type=float, default=0.5, help='weighted linear loss factor 0..1 (1 is no weighting, pure linear)')
parser.add_argument('--loss_file', type=str, default="", help='file with epoch\tloss on each line')
args = parser.parse_args()
feature_file = args.features
target_file = args.target
thresh_dB = args.thresh

feature_dim = 22
target_dim = 79
sequence_length=1
batch_size = 32
gamma = args.gamma
 
if len(args.inference) == 0:
    dataset = f32Dataset(feature_file, target_file, sequence_length, feature_dim, target_dim,num_test=args.num_test, thresh_dB=thresh_dB)
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

w1=512
class NeuralNetwork1(nn.Module):
    def __init__(self, dim, bottle_dim):
        super().__init__()
        self.bottle_dim = bottle_dim
        self.encoder = nn.Sequential(
            nn.Linear(dim, w1),
            nn.ReLU(),
            nn.Linear(w1, bottle_dim),
            nn.Tanh(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(bottle_dim, w1),
            nn.ReLU(),
            nn.Linear(w1, w1),
            nn.ReLU(),
            nn.Linear(w1, dim)
        )

    def forward(self, y):
        l = self.encoder(y)
        y_hat = self.decoder(l)
        return y_hat,l

class NeuralNetwork2(nn.Module):
    def __init__(self, dim, bottle_dim):
        super().__init__()
        self.bottle_dim = bottle_dim
        self.encoder = nn.Sequential(
            nn.Linear(dim, w1),
            nn.ReLU(),
            nn.Linear(w1, w1//2),
            nn.ReLU(),
            nn.Linear(w1//2, bottle_dim),
            nn.Tanh(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(bottle_dim, w1//2),
            nn.ReLU(),
            nn.Linear(w1//2,w1),
            nn.ReLU(),
            nn.Linear(w1, w1),
            nn.ReLU(),
            nn.Linear(w1, dim)
        )

    def forward(self, y):
        l = self.encoder(y)
        y_hat = self.decoder(l)
        return y_hat,l


match args.nn:
    case 1:
        model = NeuralNetwork1(target_dim, args.bottle_dim,).to(device)
    case 2:
        model = NeuralNetwork2(target_dim, args.bottle_dim).to(device)
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

# vanilla numpy version of weighted loss function
def my_loss_np(y_hat,y):
    w = 10**(-(1-gamma)*y)
    w = np.clip(w,None,30)
    w_dB = 20*np.log10(w)
    loss_f = ((10**y_hat - 10**y)*w)**2
    loss = np.mean(loss_f)
    return loss, loss_f, w_dB

if len(args.inference) == 0:

    # test for our custom loss function
    x = np.ones(2)
    y = 2*np.ones(2)
    w = 10**(-(1-gamma)*y)
    w = np.clip(w,None,30)
    result = my_loss(torch.from_numpy(x).to(device),torch.from_numpy(y).to(device)).cpu()
    (expected_result, tmp1, tmp2) = my_loss_np(x,y)
    if np.abs(result - expected_result) > expected_result*1E-3:
        print("my_loss() test: fail")
        print(f"my_loss(): {result} expected: {expected_result}")
        quit()
    else:
        print("my_loss() test: pass ")
    loss_fn = my_loss

    # optimizer that will be used to update weights and biases
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)
    loss_epoch=np.zeros((args.epochs))
   
    for epoch in range(args.epochs):
        sum_loss = 0.0
        for batch,(f,y) in enumerate(dataloader):
            y = y.to(device)
            y_hat,l = model(y)
            loss = loss_fn(y, y_hat)
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

        loss_epoch[epoch] = sum_loss / (batch + 1)

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
    # num_test == 0 switches off energy and V filtering, so we get all frames in test data.
    dataset_eval = f32Dataset(feature_file, target_file, sequence_length, feature_dim, target_dim, num_test=0)
    len_out = dataset_eval.__len__()
    y_hat_np = np.zeros([len_out,target_dim],dtype=np.float32)
    l_np = np.ones((len_out,args.bottle_dim),dtype=np.float32)

    with torch.no_grad():
        for f in range(len_out):
            (tmp,y) = dataset_eval.__getitem__(f)
            y_hat,l = model(torch.from_numpy(y).to(device))
            y_hat = 20*y_hat[0,].cpu().numpy()
            y_hat_np[f,] = y_hat + dataset_eval.get_edB_target(f)
            l_np[f,:] = l.cpu().numpy()

    if len(args.out_file):
        print(y_hat_np.shape)
        y_hat_np = y_hat_np.reshape((-1))
        print(y_hat_np.shape)
        y_hat_np.astype('float32').tofile(args.out_file)

    if len(args.write_latent):
        l_np = l_np.reshape((-1))
        l_np.astype('float32').tofile(args.write_latent)

# interactive frame-frame visualisation of running model on test data
if args.noplot == False:

    # we may have already loaded test data if in inference mode
    if len(args.inference) == 0:
        model.eval()
        # num_test == 0 switches off energy and V filtering, so we get all frames in test data.
        dataset_eval = f32Dataset(feature_file, target_file, sequence_length, feature_dim, target_dim, num_test=0)

    print("[click or n]-next [b]-back [j]-jump [w]-weighting [q]-quit")

    b_f_kHz = np.array([0.1998, 0.2782, 0.3635, 0.4561, 0.5569, 0.6664, 0.7855, 0.9149, 1.0556, 1.2086, 1.3749, 1.5557,
    1.7523, 1.9659, 2.1982, 2.4508, 2.7253, 3.0238, 3.3483, 3.7011])
    Fs = 8000
    Lhigh = 80
    F0high = (Fs/2)/Lhigh
    y_f_kHz = np.arange(F0high,Lhigh*F0high, F0high)/1000

    # some interesting frames male1,male1,male2,female, use 'j' to jump to them
    test_frames = np.array([61, 165, 4190, 5500])
    test_frames_ind = 0

    # called when we press a key on the plot
    akey = ''
    def on_press(event):
        global akey
        akey = event.key
    show_weighting = False

    fig, ax = plt.subplots(2, 1, height_ratios=[2, 1])
    fig.canvas.mpl_connect('key_press_event', on_press)
    ax_b = ax[0].twinx()

    with torch.no_grad():
        f = args.frame
        loop = True
        while loop:
            (tmp_,y) = dataset_eval.__getitem__(f)
            y_hat,l = model(torch.from_numpy(y).to(device))
            y_plot = 20*y[0,]
            y_hat_plot = 20*y_hat[0,].cpu().numpy()
            # TODO: compute a distortion metric like SD or MSE (linear)
            (loss, loss_f, w_dB) = my_loss_np(y_hat_plot/20,y_plot/20)
            ax[0].cla()
            t = f"f: {f}"
            ax[0].set_title(t)
            ax[0].plot(y_f_kHz,y_plot,'g')
            ax[0].plot(y_f_kHz,y_hat_plot,'r')
            ax[0].axis([0, 4, -60, 0])
            ax_b.cla()
            if show_weighting:
                ax_b.plot(y_f_kHz,w_dB,'m')
                ax_b.axis([0, 4, 0, 60])
            ax[1].cla()
            ax[1].plot(y_f_kHz,loss_f)

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
            if akey == 'w':
                show_weighting = not show_weighting
            if akey == 'q':
                loop = False
            akey = ''
