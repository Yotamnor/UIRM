# imports for the exrcise - part 1
# you can add more if you wish (but it is not really needed)
import numpy as np
import matplotlib.pyplot as plt

# pytorch imports
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
####################################################################################
"Data Loading"

transform = torchvision.transforms.ToTensor()

test_loader_e1 = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST('./datasets/', train=False, download=True,
                               transform=transform, target_transform=None), batch_size=batch_size_test, shuffle=True)

test_loader_e2 = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST('./datasets/', train=False, download=True,
                               transform=transform, target_transform=None), batch_size=batch_size_test, shuffle=True)

test_x_e1= test_loader_e1.dataset.data
test_x_e2= test_loader_e2.dataset.data

test_y_e1= test_loader_e1.dataset.targets
test_y_e2= test_loader_e2.dataset.targets

temp_test_y_e1= test_y_e1[0:int(test_y_e1.shape[0]/2)]
temp_test_y_e2= test_y_e2[int(test_y_e1.shape[0]/2):test_y_e1.shape[0]]

temp_test_x_e1 = test_x_e1[0:int(test_y_e1.shape[0]/2)]
temp_test_x_e2 = test_x_e2[int(test_y_e1.shape[0]/2):test_y_e1.shape[0]]

##############################################################################
# Disturbances

for i in range(len(temp_test_x_e1)):
    changePa = np.random.rand(1)
    if (changePa < 1):
        temp_test_x_e1[i, 0:7, 0:7] = 255 # for e_s in E_train
        # temp_test_x_e1[i, 0:7, -7:-1] =  255 # for e_s in E_test

for i in range(len(temp_test_x_e2)):
    # changePa = np.random.rand(1)
    if (changePa < 1):
        temp_test_x_e2[i, -7:-1, -7:-1] = 255 # for e_s in E_train
        # temp_test_x_e2[i, -7:-1, 0:7] = 255 # for e_s in E_test


#############################################################################
"Data Orginization"

temp_test_x_e1 = temp_test_x_e1
temp_test_x_e2 = temp_test_x_e2
temp_dataset_e1 = torch.utils.data.TensorDataset(temp_test_x_e1.unsqueeze(1).float()/255, temp_test_y_e1)
# temp_dataset_e1 = torch.utils.data.TensorDataset(temp_test_x_e1.float(), temp_test_y_e1)
temp_dataset_e1.data = temp_test_x_e1
temp_dataset_e1.targets = temp_test_y_e1
temp_dataset_e2 = torch.utils.data.TensorDataset(temp_test_x_e2.unsqueeze(1).float()/255, temp_test_y_e2)
# temp_dataset_e2 = torch.utils.data.TensorDataset(temp_test_x_e2.float(), temp_test_y_e2)
temp_dataset_e2.data = temp_test_x_e2
temp_dataset_e2.targets = temp_test_y_e2

test_loader_e1= DataLoader(temp_dataset_e1, batch_size=batch_size_test, shuffle=False)
test_loader_e2= DataLoader(temp_dataset_e2, batch_size=batch_size_test, shuffle=False)

test_x_e1= test_loader_e1.dataset.data
test_x_e2= test_loader_e2.dataset.data

########################################################################################################
"Parameters"

HIDDEN_SIZE=256
X_DIM=28*28
Z_DIM=10
if torch.cuda.is_available():
    torch.cuda.current_device()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
vae = Vae_Irm(x_dim=X_DIM, z_dim=Z_DIM, hidden_size=HIDDEN_SIZE, device=device).to(device)

###########################################################################################################
"Examples"

torch.Size([1000, 1, 28, 28])
zero_channel = torch.zeros((28,28,1))

fig, axes = plt.subplots(1, 2, figsize=(15, 5))  # Create a figure with 1 row and 2 columns
# Plot for e=1
# to_plot_e1 = torch.cat((train_loader_e1.dataset.data[6,:,:].permute(1, 2, 0), zero_channel), dim=2)
to_plot_e1 = test_loader_e1.dataset.data[6, :, :].squeeze()
axes[0].imshow(to_plot_e1, cmap='gray')
axes[0].axes.get_xaxis().set_ticks([])
axes[0].axes.get_yaxis().set_ticks([])
axes[0].set_title("e=3")  # Add label for e=1
#
# Plot for e=2
# to_plot_e2 = torch.cat((train_loader_e2.dataset.data[6,:,:].permute(1, 2, 0), zero_channel), dim=2)
to_plot_e2 = test_loader_e2.dataset.data[6, :, :].squeeze()
axes[1].imshow(to_plot_e2, cmap='gray')
axes[1].axes.get_xaxis().set_ticks([])
axes[1].axes.get_yaxis().set_ticks([])
axes[1].set_title("e=4")  # Add label for e=2
plt.tight_layout()  # Adjust layout to avoid overlap
plt.show()
fig.savefig('Testenv_combined.png')  # Save the combined figure
###############################################################################################################################
'Training Load'
train_loader_e1 = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST('./datasets/', train=True, download=True,
                               transform=transform, target_transform=None), batch_size=batch_size_train, shuffle=True)
train_loader_e2 = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST('./datasets/', train=True, download=True,
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
# Disturbances
#
for i in range(len(temp_train_x_e1)):
    changePa = np.random.rand(1)
    if (changePa < 1):
        'SMNIST'
        temp_train_x_e1[i, 0:7, 0:7] = 255
        'RMNIST'

for i in range(len(temp_train_x_e2)):
    # changePa = np.random.rand(1)
    if (changePa < 1):
        'SMNIST'
        temp_train_x_e2[i, -7:-1, -7:-1] = 255
        'RMNIST'

#############################################################################
"Data Organization"

temp_train_x_e1 = temp_train_x_e1.unsqueeze(1)
temp_train_x_e2 = temp_train_x_e2.unsqueeze(1)

temp_dataset_e1 = torch.utils.data.TensorDataset(temp_train_x_e1.float()/255, temp_train_y_e1)
temp_dataset_e1.data = temp_train_x_e1.unsqueeze(1).float()/255
temp_dataset_e1.targets = temp_train_y_e1
temp_dataset_e2 = torch.utils.data.TensorDataset(temp_train_x_e2.float()/255, temp_train_y_e2)
temp_dataset_e2.data = temp_train_x_e2.unsqueeze(1).float()/255
temp_dataset_e2.targets = temp_train_y_e2
#
# from torchvision import transforms
train_loader_e1= DataLoader(temp_dataset_e1, batch_size=batch_size_train, shuffle=True)
train_loader_e2= DataLoader(temp_dataset_e2, batch_size=batch_size_train, shuffle=True)

###########################################################################################################################
'Fine Tune!- Turned off'

vae.load_state_dict(torch.load('vae_irm.pth'))
vae.requires_grad_(True)
# vae.encoder_env2.load_state_dict(vae.encoder_env1.state_dict())
vae.encoder_causal.requires_grad_(False)
vae.decoder.requires_grad_(False)

# vae,train_recon_errors,train_kls,train_losses= train_beta_vae(1, train_loader_e1, train_loader_e2, NUM_EPOCHS=5, vae=vae, fine_tune=1)
######################################################################################################
"Inference"
test_x_e1 = test_loader_e1.dataset.data
test_y_e1= test_loader_e1.dataset.targets
test_e1 = enumerate(test_loader_e1)
batch_idx_e1, (test_data_e1, test_targets_e1) = next(test_e1)

# test_data_e1.shape
#
# torch.Size([1000, 1, 28, 28])

test_x_e2 = test_loader_e2.dataset.data
test_y_e2= test_loader_e2.dataset.targets
test_e2 = enumerate(test_loader_e2)
batch_idx_e2, (test_data_e2, test_targets_e2) = next(test_e2)
#
# test_data_e2.shape
#
# torch.Size([1000, 1, 28, 28])
"Plots"
import matplotlib.pyplot as plt
########################################################################
"Org"
fig = plt.figure(figsize=(9, 5))
for i in range(3):
    plt.subplot(2,3,i+1)
    plt.tight_layout()
    x_e = test_x_e1[i+10].unsqueeze(0).unsqueeze(0).to(device).float()/255
    to_plot = x_e.cpu().detach().squeeze()#.permute(1, 2, 0)
    plt.imshow(to_plot, cmap='gray')
    plt.title("Label: {}, e source: 1".format(test_y_e1[i+10]))
    plt.xticks([])
    plt.yticks([])
for i in range(3):
    plt.subplot(2,3,i+4)
    plt.tight_layout()
    x_e = test_x_e2[i+10].unsqueeze(0).unsqueeze(0).to(device).float()/255
    to_plot = x_e.cpu().detach().squeeze()#.permute(1, 2, 0)
    plt.imshow(to_plot, cmap='gray')
    plt.title("Label: {}, e source: 2".format(test_y_e2[i+10]))
    plt.xticks([])
    plt.yticks([])
fig
plt.show()
#####################################################################
"Inf!"

fig = plt.figure(figsize=(9, 5))
for i in range(3):
    plt.subplot(2,3,i+1)
    plt.tight_layout()
    # x_e = test_x_e2[i+15].view(-1, X_DIM).to(device).float()/255
    x_e = test_x_e1[i+10].unsqueeze(0).unsqueeze(0).to(device).float()/255
    z, mu, logvar = vae.encode(x_e,1)
    z_e = torch.randn(1, 10)
    z_e[0,8] = z_e[0,8]+100
    z[0,10:20] = z_e.to(device)
    # z_e = torch.randn(1, 10)
    # z_e[0,8] = z_e[0,8]+100
    ##
    # z_e12 = torch.randn(1, 10)
    # z_e12[0, 8] = z_e12[0, 8] + 50
    # z_e12[0, 9] = z_e12[0, 9] + 50
    # z_c, mu_c, logvar_c = vae.encoder_causal(x_e.to(device), z_e12.to(device))
    # z = torch.cat((z_c, z_e.to(device)), dim=1)
    ##
    z2, mu2, logvar2 = vae.encode(x_e, 2)
    z2[0, 10:20] = z_e.to(device)
    z = (z + z2) / 2
    ##
    x = vae.decode(z)
    # plt.imshow(x.cpu().view(28,28).detach(), cmap='gray', interpolation='none')
    # to_plot = torch.cat((x_e.cpu().detach().squeeze().permute(1, 2, 0), zero_channel), dim=2)
    to_plot = x.cpu().detach().squeeze()#.permute(1, 2, 0)
    plt.imshow(to_plot, cmap='gray')
    plt.title("Label: {}, e source: 1".format(test_y_e1[i+10]))
    plt.xticks([])
    plt.yticks([])
for i in range(3):
    plt.subplot(2,3,i+4)
    plt.tight_layout()
    # x_e = test_x_e2[i+15].view(-1, X_DIM).to(device).float()/255
    x_e = test_x_e2[i+10].unsqueeze(0).unsqueeze(0).to(device).float()/255
    z, mu, logvar = vae.encode(x_e,2)
    z_e = torch.randn(1, 10)
    z_e[0,8] = z_e[0,8]+100
    z[0,10:20] = z_e.to(device)
    # z_e = torch.randn(1, 10)
    # z_e[0,8] = z_e[0,8]+100
    # z_c, mu_c, logvar_c = vae.encoder_causal(x_e.to(device), z_e.to(device))
    # z = torch.cat((z_c, z_e.to(device)), dim=1)
    ##
    z2, mu2, logvar2 = vae.encode(x_e, 2)
    z2[0, 10:20] = z_e.to(device)
    z = (z + z2) / 2
    ##
    x = vae.decode(z)
    # plt.imshow(x.cpu().view(28,28).detach(), cmap='gray', interpolation='none')
    # to_plot = torch.cat((x_e.cpu().detach().squeeze().permute(1, 2, 0), zero_channel), dim=2)
    to_plot = x.cpu().detach().squeeze()#.permute(1, 2, 0)
    plt.imshow(to_plot, cmap='gray')
    plt.title("Label: {}, e source: 2".format(test_y_e2[i+10]))
    plt.xticks([])
    plt.yticks([])
fig
plt.show()