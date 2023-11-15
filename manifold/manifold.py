"""
Manifold ML experiment - training script.

Train a model to generate low pitched speech with good time domain energy distribution over a
pitch cycle.

make -f ratek_resampler.mk train_120_b20_ml.f32 train_120_y80_ml.f32
python3 manifold.py train_120_b20_ml.f32 train_120_y80_ml.f32

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
                edB_target = 10*np.log10(e)
                self.targets[j,] -= edB_target
                #print(self.targets[j,])
 
                # b and y vectors are in x_dB = 20*log10(x), scale down to log10(x).  We don't need to scale
                # Wo and voicing (last two floats in feature vector)
                self.features[j,:20] = self.features[j,:20]/20
                self.targets[j,] = self.targets[j,]/20

                j += 1
        self.num_sequences = j
        if num_test:
            print(f"Training: First {num_test} reserved for testing, {self.num_sequences}/{num_sequences_in} voiced vectors with energy > {thresh_dB} used")
        else:
            print(f"Testing: {self.num_sequences} loaded")

    def __len__(self):
        return self.num_sequences

    def __getitem__(self, index):
        #print(self.features[1000,:])
        #quit()
        features = self.features[index * self.sequence_length: (index + 1) * self.sequence_length, :]
        targets = self.targets[index * self.sequence_length: (index + 1) * self.sequence_length, :]
        return features, targets
    
parser = argparse.ArgumentParser()
parser.add_argument('features', type=str, help='path to feature file [b[22] Wo v] .f32 format')
parser.add_argument('target', type=str, help='path to target file [y[79]] in .f32 format')
parser.add_argument('--frame', type=int, default=165, help='frame # to start viewing')
parser.add_argument('--noplot', action='store_true', help='disable plots after training')
parser.add_argument('--num_test', type=int, default=60*100, help='number of vectors reserved for testing')
parser.add_argument('--epochs', type=int, default=10, help='number of training epochs')
parser.add_argument('--thresh', type=float, default=10, help='energy threshold for training data in dB')
args = parser.parse_args()
feature_file = args.features
target_file = args.target
thresh_dB = args.thresh

feature_dim = 22
target_dim = 79
sequence_length=1
batch_size = 32

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

# Define model
w1=512
class NeuralNetwork1(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(input_dim, w1),
            nn.ReLU(),
            nn.Linear(w1, w1),
            nn.ReLU(),
            nn.Linear(w1, output_dim)
       )

    def forward(self, x):
        y = self.linear_relu_stack(x)
        return y

nf=32
w2=128
class NeuralNetwork2(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.c1 = nn.Conv1d(1, nf, 5)
        self.f1 = nn.Flatten()
        self.r1 = nn.ReLU()
        self.l1 = nn.Linear(nf*(input_dim-4), w2)
        self.r2 = nn.ReLU()
        self.l2 = nn.Linear(w2, output_dim)

    def forward(self, x):
        #print(x.shape)
        # TODO ignore Wo and voicing
        x = self.c1(x)
        #print(x.shape)
        x = self.f1(x)
        #print(x.shape)
        x = self.r1(x)
        x = self.l1(x)
        x = self.r2(x)
        x = self.l2(x)
        return x

class NeuralNetwork3(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(input_dim, 40),
            nn.ReLU(),
            nn.Linear(40, output_dim),
            nn.ReLU(),
            nn.Linear(output_dim, output_dim)
       )

    def forward(self, x):
        y = self.linear_relu_stack(x)
        return y

model = NeuralNetwork1(feature_dim, target_dim).to(device)

print(model)
num_weights = sum(p.numel() for p in model.parameters())
print(f"weights: {num_weights} float32 memory: {num_weights*4}")

# prototype custom loss function
def my_loss_mse(y_hat, y):
    loss = torch.mean((y_hat - y)**2)
    return loss
 
gamma = 0.5
 
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
optimizer = torch.optim.SGD(model.parameters(), lr=5E-2)

for epoch in range(args.epochs):
    sum_loss = 0.0
    for batch,(f,y) in enumerate(dataloader):
        #print(f,y)
        f = f.to(device)
        y = y.to(device)
        y_hat = model(f)
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

if args.noplot:
    quit()

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
    b = args.frame
    loop = True
    while loop:
        (f,y) = dataset_eval.__getitem__(b)
        y_hat = model(torch.from_numpy(f).to(device))
        f_plot = 20*f[0,]
        y_plot = 20*y[0,]
        y_hat_plot = 20*y_hat[0,].cpu().numpy()
        # TODO: compute a distortion metric like SD or MSE (linear)
        (loss, loss_f, w_dB) = my_loss_np(y_hat_plot/20,y_plot/20)
        ax[0].cla()
        ax[0].plot(b_f_kHz,f_plot[0:20])
        t = f"f: {b}"
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
            b -= 1
        if akey == 'n' or button == False:
            b += 1
        if akey == 'j':
            b = test_frames[test_frames_ind]
            test_frames_ind += 1
            test_frames_ind = np.mod(test_frames_ind,4)
        if akey == 'w':
            show_weighting = not show_weighting
        if akey == 'q':
            loop = False
        akey = ''
