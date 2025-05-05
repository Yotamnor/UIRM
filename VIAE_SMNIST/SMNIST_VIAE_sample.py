import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader, Dataset, ConcatDataset
import torchvision

from SMNIST_VIAE_model import train_beta_vae, Vae_Irm


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
num_of_channels = 1

###############################################################
"Deterministic Settings"
# random_seed = 1
# torch.backends.cudnn.enabled = False
# torch.manual_seed(random_seed)
# ##################################################################

"Data Loading"

transform = torchvision.transforms.ToTensor()


train_loader_e1 = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST('./datasets/', train=True, download=True,
                               transform=transform, target_transform=None), batch_size=batch_size_train, shuffle=True)
train_loader_e2 = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST('./datasets/', train=True, download=True,
                               transform=transform, target_transform=None), batch_size=batch_size_train, shuffle=True)

test_loader = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST('./datasets/', train=False, download=True,
                               transform=transform, target_transform=None), batch_size=batch_size_train, shuffle=True)

train_x_e1= train_loader_e1.dataset.data
train_x_e2= train_loader_e2.dataset.data

train_y_e1= train_loader_e1.dataset.targets
train_y_e2= train_loader_e2.dataset.targets

temp_train_y_e1= train_y_e1[0:int(train_y_e1.shape[0]/2)]
temp_train_y_e2= train_y_e2[int(train_y_e1.shape[0]/2):train_y_e1.shape[0]]

temp_train_x_e1 = train_x_e1[0:int(train_y_e1.shape[0]/2)]
temp_train_x_e2 = train_x_e2[int(train_y_e1.shape[0]/2):train_y_e1.shape[0]]

##############################################################################
# # Disturbances
#
for i in range(len(temp_train_x_e1)):
    changePa = np.random.rand(1)
    if (changePa < 1):
        temp_train_x_e1[i, 0:7, 0:7] = 255
        'RMNIST'
        # temp_train_x_e1[i, :, :] = torchvision.transforms.functional.rotate(temp_train_x_e1[i, :, :].unsqueeze(0).unsqueeze(0), angle=45)


for i in range(len(temp_train_x_e2)):
    # changePa = np.random.rand(1)
    if (changePa < 1):
        temp_train_x_e2[i, -7:-1, -7:-1] = 255
        'RMNIST'
        # temp_train_x_e2[i, :, :] = torchvision.transforms.functional.rotate(temp_train_x_e2[i, :, :].unsqueeze(0).unsqueeze(0), angle=315)
#############################################################################
"Re-Organize"

temp_dataset_e1 = torch.utils.data.TensorDataset(temp_train_x_e1.float()/255, temp_train_y_e1)
temp_dataset_e1.data = temp_train_x_e1.unsqueeze(1).float()/255
temp_dataset_e1.targets = temp_train_y_e1
temp_dataset_e2 = torch.utils.data.TensorDataset(temp_train_x_e2.float()/255, temp_train_y_e2)
temp_dataset_e2.data = temp_train_x_e2.unsqueeze(1).float()/255
temp_dataset_e2.targets = temp_train_y_e2

from torchvision import transforms
train_loader_e1= DataLoader(temp_dataset_e1, batch_size=batch_size_train, shuffle=True)
train_loader_e2= DataLoader(temp_dataset_e2, batch_size=batch_size_train, shuffle=True)

###################################################################################################
"Plot Examples"

fig = plt.figure(figsize=(10,4))
ax = fig.add_subplot(1,1,1)
# to_plot = torch.cat((train_loader_e2.dataset.data[1,:,:].permute(1, 2, 0), zero_channel), dim=2)
to_plot = train_loader_e2.dataset.data[6,:,:].squeeze()#.permute(1, 2, 0)
ax.imshow(to_plot, cmap='gray')
ax.axes.get_xaxis().set_ticks([])
ax.axes.get_yaxis().set_ticks([])
plt.show()

fig = plt.figure(figsize=(10,4))
ax = fig.add_subplot(1,1,1)
# to_plot = torch.cat((train_loader_e1.dataset.data[1,:,:].permute(1, 2, 0), zero_channel), dim=2)
to_plot = train_loader_e1.dataset.data[6,:,:].squeeze()#.permute(1, 2, 0)
ax.imshow(to_plot, cmap='gray')
ax.axes.get_xaxis().set_ticks([])
ax.axes.get_yaxis().set_ticks([])
plt.show()

fig = plt.figure(figsize=(10,4))
for j in range(10):
    ax = fig.add_subplot(3,10, j+1)
    # to_plot = np.concatenate((train_loader_e1.dataset.data[j,:,:].permute(1, 2, 0), zero_channel), axis=2)
    to_plot = train_loader_e1.dataset.data[j,:,:].squeeze()#.permute(1, 2, 0)
    ax.imshow(to_plot, cmap='gray')
    ax.axes.get_xaxis().set_ticks([])
    ax.axes.get_yaxis().set_ticks([])
    if j==0:
        ax.set_title("Environment 1")

for j in range(10):
    ax = fig.add_subplot(3,10, 10 + j+1)
    # to_plot = np.concatenate((train_loader_e2.dataset.data[j,:,:].permute(1, 2, 0), zero_channel), axis=2)
    to_plot = train_loader_e2.dataset.data[j,:,:].squeeze()#.permute(1, 2, 0)
    ax.imshow(to_plot, cmap='gray')
    ax.axes.get_xaxis().set_ticks([])
    ax.axes.get_yaxis().set_ticks([])
    if j==0:
        ax.set_title("Environment 2")
plt.show()


HIDDEN_SIZE=256
X_DIM=28*28
Z_DIM=10
if torch.cuda.is_available():
    torch.cuda.current_device()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
vae = Vae_Irm(x_dim=X_DIM, z_dim=Z_DIM, hidden_size=HIDDEN_SIZE, device=device).to(device)
vae.load_state_dict(torch.load('vae_irm.pth'))


n_samples=5
fig = plt.figure(figsize=(7,7))
bias_e1 = torch.zeros(20)
bias_e1[18]=100
# samples=np.reshape((vae.sample(n_samples, bias = bias_e1)).data.cpu().numpy(),(n_samples,28,28))
samples=np.reshape((vae.sample(n_samples, bias = bias_e1)).data.cpu().numpy(),(n_samples,num_of_channels
                                                                               ,28,28))
for j in range(n_samples):
    ax = fig.add_subplot(3,n_samples, j+1)
    # to_plot = np.concatenate((samples[j].transpose(1, 2, 0), zero_channel), axis=2)
    to_plot = samples[j].squeeze()#.transpose(1, 2, 0)
    ax.imshow(to_plot, cmap='gray')
    ax.axes.get_xaxis().set_ticks([])
    ax.axes.get_yaxis().set_ticks([])
    if j==0:
        ax.set_title("Environment 1")

bias_e2 = torch.zeros(20)
bias_e2[19]=100
samples=np.reshape((vae.sample(n_samples, bias = bias_e2)).data.cpu().numpy(),(n_samples,num_of_channels,28,28))
for j in range(n_samples):
    ax = fig.add_subplot(3,n_samples, n_samples + j+1)
    # to_plot = np.concatenate((samples[j].transpose(1, 2, 0), zero_channel), axis=2)
    to_plot = samples[j].squeeze()#.transpose(1, 2, 0)
    ax.imshow(to_plot, cmap='gray')
    ax.axes.get_xaxis().set_ticks([])
    ax.axes.get_yaxis().set_ticks([])
    if j==0:
        ax.set_title("Environment 2")

bias_I = torch.zeros(20)
# bias_I[0:9] = torch.ones(10)
samples=np.reshape((vae.sample(n_samples, bias = bias_I)).data.cpu().numpy(),(n_samples,num_of_channels,28,28))
for j in range(n_samples):
    ax = fig.add_subplot(3,n_samples, 2*n_samples + j+1)
    # to_plot = np.concatenate((samples[j].transpose(1, 2, 0), zero_channel), axis=2)
    to_plot = samples[j].squeeze()#.transpose(1, 2, 0)
    ax.imshow(to_plot, cmap='gray')
    ax.axes.get_xaxis().set_ticks([])
    ax.axes.get_yaxis().set_ticks([])
    if j==0:
        ax.set_title("Invariant")

plt.show()

# Let's create with the same Z_C!

z_c = torch.randn(1, 10)
n_samples=5

fig = plt.figure(figsize=(7,7))
bias_e1 = torch.zeros(20)
bias_e1[18]=100

z_e = torch.randn(n_samples, 10)
z_c_repeated = z_c.repeat(n_samples, 1)
z_pre = torch.cat((z_c_repeated, z_e), dim=1)
z_pre = z_pre + bias_e1

samples=np.reshape((vae.sample(n_samples, bias = bias_e1, freeze=2, z_pre = z_pre)).data.cpu().numpy(),(n_samples,num_of_channels,28,28))
for j in range(n_samples):
    ax = fig.add_subplot(3,n_samples, j+1)
    # to_plot = to_plot = np.concatenate((samples[j].transpose(1, 2, 0), zero_channel), axis=2)
    to_plot = samples[j].squeeze()#.transpose(1, 2, 0)
    ax.imshow(to_plot, cmap='gray')
    ax.axes.get_xaxis().set_ticks([])
    ax.axes.get_yaxis().set_ticks([])
    if j==0:
        ax.set_title("Environment 1")

bias_e2 = torch.zeros(20)
bias_e2[19]=100

z_e = torch.randn(n_samples, 10)
z_c_repeated = z_c.repeat(n_samples, 1)
z_pre = torch.cat((z_c_repeated, z_e), dim=1)
z_pre = z_pre + bias_e2

samples=np.reshape((vae.sample(n_samples, bias = bias_e2, freeze=2, z_pre=z_pre)).data.cpu().numpy(),(n_samples,num_of_channels,28,28))
for j in range(n_samples):
    ax = fig.add_subplot(3,n_samples, n_samples + j+1)
    # to_plot = np.concatenate((samples[j].transpose(1, 2, 0), zero_channel), axis=2)
    to_plot = samples[j].squeeze()#.transpose(1, 2, 0)
    ax.imshow(to_plot, cmap='gray')
    ax.axes.get_xaxis().set_ticks([])
    ax.axes.get_yaxis().set_ticks([])
    if j==0:
        ax.set_title("Environment 2")
plt.show()

#####################################################################################################
# Let's create average sample

z_c = torch.randn(1, 10)

n_samples=1000

bias_e1 = torch.zeros(20)
bias_e1[18]=100

z_e = torch.randn(n_samples, 10)
z_c_repeated = z_c.repeat(n_samples, 1)
z_pre = torch.cat((z_c_repeated, z_e), dim=1)
z_pre = z_pre + bias_e1

samples_e1=np.reshape((vae.sample(n_samples, bias = bias_e1, freeze=2, z_pre = z_pre)).data.cpu().numpy(),(n_samples,num_of_channels,28,28))

bias_e2 = torch.zeros(20)
bias_e2[19]=100

z_e = torch.randn(n_samples, 10)
z_c_repeated = z_c.repeat(n_samples, 1)
z_pre = torch.cat((z_c_repeated, z_e), dim=1)
z_pre = z_pre + bias_e2

samples_e2=np.reshape((vae.sample(n_samples, bias = bias_e2, freeze=2, z_pre=z_pre)).data.cpu().numpy(),(n_samples,num_of_channels,28,28))

averaged_samples_tmp = np.mean([samples_e1, samples_e2], axis=0)
averaged_samples = np.mean(averaged_samples_tmp, axis=0)

fig = plt.figure(figsize=(7,7))
ax = fig.add_subplot(1,1,1)
# to_plot = np.concatenate((averaged_samples.transpose(1, 2, 0), zero_channel), axis=2)
to_plot = averaged_samples.squeeze()#.transpose(1, 2, 0)
ax.imshow(to_plot, cmap='gray')
ax.axes.get_xaxis().set_ticks([])
ax.axes.get_yaxis().set_ticks([])
ax.set_title("Avg Sample")
plt.show()
