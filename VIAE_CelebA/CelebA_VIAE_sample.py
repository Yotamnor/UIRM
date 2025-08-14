import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader, Dataset, ConcatDataset
import torchvision

from CelebA_VIAE_model import train_beta_vae, Vae_Irm


datapath = '/files/'
resultsPath = 'C:/Users/Yotam/.spyder-py3/MINST_test/results/'

n_epochs = 50
batch_size_train = 128#64
batch_size_test = 128
learning_rate = 1e-3
momentum = 0.5
log_interval = 10
lam = 50
w_dis_history = np.array([])
acc_vec = np.array([])
num_of_channels = 3

###############################################################
"Deterministic Settings"
# random_seed = 1
# torch.backends.cudnn.enabled = False
# torch.manual_seed(random_seed)
###########################################################################################

from torchvision.datasets import CelebA
import torchvision.transforms as transforms

# Set where to store the data
data_root = "/home/yotamnor/files/Yotam_env/VAE_VAE/CelebA_VIAE/data" #"./data/"#

# Define any transforms (resize, normalize, etc.)
transform = transforms.Compose([
    transforms.Resize(64),
    transforms.CenterCrop(64),
    transforms.ToTensor()
])

celeba_train_dataset = torchvision.datasets.ImageFolder(root=data_root, transform=transform)

# Gender attribute is at index 20: 1 = Male, -1 = Female
import pandas as pd
all_attributes = pd.read_csv("./data/celeba/list_attr_celeba.csv")
# all_attributes = celeba_train_dataset.attr  # shape: (N, 40)
male_indices = torch.where(torch.tensor(all_attributes['Male'] == 1))#[0]
female_indices = torch.where(torch.tensor(all_attributes['Male'] == -1))#[0]


male_subset = torch.utils.data.Subset(celeba_train_dataset, male_indices[0].tolist())
female_subset = torch.utils.data.Subset(celeba_train_dataset, female_indices[0].tolist())


# Optional: Wrap in a DataLoader
celeba_x_train_e1 = male_subset #DataLoader(male_subset, batch_size=batch_size_train, shuffle=True)
celeba_x_train_e2 = female_subset#DataLoader(female_subset, batch_size=batch_size_train, shuffle=True)


# celeba_train_e1 = celeba_x_train_e1.dataset[male_indices[0]]
#
# celeba_train_e2 = celeba_x_train_e2.dataset[female_indices[0]]
celeba_train_e1 = celeba_train_dataset[male_indices[0].tolist()[0]]
# celeba_train_e2 = [celeba_train_dataset[i] for i in female_indices[0].tolist()]
from torch.utils.data import DataLoader, Subset

# Set batch size for faster loading
batch_size = 512  # Adjust based on your available memory

# Create a DataLoader for your subsetssbssss
subs1 = Subset(celeba_train_dataset, male_indices[0].tolist())
subs2 = Subset(celeba_train_dataset, female_indices[0].tolist())

celeba_x_train_e1 = DataLoader(subs1, batch_size=batch_size_train, shuffle=True)
celeba_x_train_e2 = DataLoader(subs2, batch_size=batch_size_train, shuffle=True)
###############################################################################

fig = plt.figure(figsize=(10,4))
ax = fig.add_subplot(1,1,1)
# to_plot = torch.cat((train_loader_e2.dataset.data[1,:,:].permute(1, 2, 0), zero_channel), dim=2)
to_plot = celeba_x_train_e2.dataset[6][0].permute(1, 2, 0)
ax.imshow(to_plot)
ax.axes.get_xaxis().set_ticks([])
ax.axes.get_yaxis().set_ticks([])
plt.show()

fig = plt.figure(figsize=(10,4))
ax = fig.add_subplot(1,1,1)
# to_plot = torch.cat((train_loader_e1.dataset.data[1,:,:].permute(1, 2, 0), zero_channel), dim=2)
to_plot = celeba_x_train_e1.dataset[6][0].permute(1, 2, 0)
ax.imshow(to_plot)
ax.axes.get_xaxis().set_ticks([])
ax.axes.get_yaxis().set_ticks([])
plt.show()

fig = plt.figure(figsize=(10,4))
for j in range(10):
    ax = fig.add_subplot(3,10, j+1)
    # to_plot = np.concatenate((train_loader_e1.dataset.data[j,:,:].permute(1, 2, 0), zero_channel), axis=2)
    to_plot = celeba_x_train_e1.dataset[j][0].permute(1, 2, 0)
    ax.imshow(to_plot)
    ax.axes.get_xaxis().set_ticks([])
    ax.axes.get_yaxis().set_ticks([])
    if j==0:
        ax.set_title("Environment 1")

for j in range(10):
    ax = fig.add_subplot(3,10, 10 + j+1)
    # to_plot = np.concatenate((train_loader_e2.dataset.data[j,:,:].permute(1, 2, 0), zero_channel), axis=2)
    to_plot = celeba_x_train_e2.dataset[j][0].permute(1, 2, 0)
    ax.imshow(to_plot)
    ax.axes.get_xaxis().set_ticks([])
    ax.axes.get_yaxis().set_ticks([])
    if j==0:
        ax.set_title("Environment 2")

plt.show()
######################################################################################
HIDDEN_SIZE=256
X_DIM=28*28
Z_C_DIM=56

Z_E_DIM=8
if torch.cuda.is_available():
    torch.cuda.current_device()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
vae = Vae_Irm(z_c_dim=Z_C_DIM,z_e_dim=Z_E_DIM, device=device).to(device)
vae.load_state_dict(torch.load('vae_irm.pth'))

n_samples=5
fig = plt.figure(figsize=(7,7))
bias_e1 = torch.zeros(64)
bias_e1[62]=100
# samples=np.reshape((vae.sample(n_samples, bias = bias_e1)).data.cpu().numpy(),(n_samples,28,28))
samples=np.reshape((vae.sample(n_samples, bias = bias_e1)).data.cpu().numpy(),(n_samples,num_of_channels
                                                                               ,64,64))
for j in range(n_samples):
    ax = fig.add_subplot(3,n_samples, j+1)
    # to_plot = np.concatenate((samples[j].transpose(1, 2, 0), zero_channel), axis=2)
    to_plot = samples[j].transpose(1, 2, 0)
    ax.imshow(to_plot)
    ax.axes.get_xaxis().set_ticks([])
    ax.axes.get_yaxis().set_ticks([])
    if j==0:
        ax.set_title("Environment 1")

bias_e2 = torch.zeros(64)
bias_e2[63]=100
samples=np.reshape((vae.sample(n_samples, bias = bias_e2)).data.cpu().numpy(),(n_samples,num_of_channels,64,64))
for j in range(n_samples):
    ax = fig.add_subplot(3,n_samples, n_samples + j+1)
    # to_plot = np.concatenate((samples[j].transpose(1, 2, 0), zero_channel), axis=2)
    to_plot = samples[j].transpose(1, 2, 0)
    ax.imshow(to_plot)
    ax.axes.get_xaxis().set_ticks([])
    ax.axes.get_yaxis().set_ticks([])
    if j==0:
        ax.set_title("Environment 2")

bias_I = torch.zeros(64)
# bias_I[0:9] = torch.ones(10)
samples=np.reshape((vae.sample(n_samples, bias = bias_I)).data.cpu().numpy(),(n_samples,num_of_channels,64,64))
for j in range(n_samples):
    ax = fig.add_subplot(3,n_samples, 2*n_samples + j+1)
    # to_plot = np.concatenate((samples[j].transpose(1, 2, 0), zero_channel), axis=2)
    to_plot = samples[j].transpose(1, 2, 0)
    ax.imshow(to_plot, cmap='gray')
    ax.axes.get_xaxis().set_ticks([])
    ax.axes.get_yaxis().set_ticks([])
    if j==0:
        ax.set_title("Invariant")

plt.show()

# Let's create with the same Z_C!

z_c = torch.randn(1, 56)
n_samples=5

fig = plt.figure(figsize=(7,7))
bias_e1 = torch.zeros(64)
bias_e1[62]=100

z_e = torch.randn(n_samples, 8)
z_c_repeated = z_c.repeat(n_samples, 1)
z_pre = torch.cat((z_c_repeated, z_e), dim=1)
z_pre = z_pre + bias_e1

samples=np.reshape((vae.sample(n_samples, bias = bias_e1, freeze=2, z_pre = z_pre)).data.cpu().numpy(),(n_samples,num_of_channels,64,64))
for j in range(n_samples):
    ax = fig.add_subplot(3,n_samples, j+1)
    # to_plot = to_plot = np.concatenate((samples[j].transpose(1, 2, 0), zero_channel), axis=2)
    to_plot = samples[j].transpose(1, 2, 0)
    ax.imshow(to_plot)
    ax.axes.get_xaxis().set_ticks([])
    ax.axes.get_yaxis().set_ticks([])
    if j==0:
        ax.set_title("Environment 1")

bias_e2 = torch.zeros(64)
bias_e2[63]=100

z_e = torch.randn(n_samples, 8)
z_c_repeated = z_c.repeat(n_samples, 1)
z_pre = torch.cat((z_c_repeated, z_e), dim=1)
z_pre = z_pre + bias_e2

samples=np.reshape((vae.sample(n_samples, bias = bias_e2, freeze=2, z_pre=z_pre)).data.cpu().numpy(),(n_samples,num_of_channels,64,64))
for j in range(n_samples):
    ax = fig.add_subplot(3,n_samples, n_samples + j+1)
    # to_plot = np.concatenate((samples[j].transpose(1, 2, 0), zero_channel), axis=2)
    to_plot = samples[j].transpose(1, 2, 0)
    ax.imshow(to_plot)
    ax.axes.get_xaxis().set_ticks([])
    ax.axes.get_yaxis().set_ticks([])
    if j==0:
        ax.set_title("Environment 2")

plt.show()
####################################################################################################

# Let's create average sample

z_c = torch.randn(1,56)

n_samples=100

bias_e1 = torch.zeros(64)
bias_e1[62]=100

z_e = torch.randn(n_samples, 8)
z_c_repeated = z_c.repeat(n_samples, 1)
z_pre = torch.cat((z_c_repeated, z_e), dim=1)
z_pre = z_pre + bias_e1

samples_e1=np.reshape((vae.sample(n_samples, bias = bias_e1, freeze=2, z_pre = z_pre)).data.cpu().numpy(),(n_samples,num_of_channels,64,64))

bias_e2 = torch.zeros(64)
bias_e2[63]=100

z_e = torch.randn(n_samples, 8)
z_c_repeated = z_c.repeat(n_samples, 1)
z_pre = torch.cat((z_c_repeated, z_e), dim=1)
z_pre = z_pre + bias_e2

samples_e2=np.reshape((vae.sample(n_samples, bias = bias_e2, freeze=2, z_pre=z_pre)).data.cpu().numpy(),(n_samples,num_of_channels,64,64))

averaged_samples_tmp = np.mean([samples_e1, samples_e2], axis=0)
averaged_samples = np.mean(averaged_samples_tmp, axis=0)

fig = plt.figure(figsize=(7,7))
ax = fig.add_subplot(1,1,1)
# to_plot = np.concatenate((averaged_samples.transpose(1, 2, 0), zero_channel), axis=2)
to_plot = averaged_samples.transpose(1, 2, 0)
ax.imshow(to_plot)
ax.axes.get_xaxis().set_ticks([])
ax.axes.get_yaxis().set_ticks([])
ax.set_title("Avg Sample")
plt.show()