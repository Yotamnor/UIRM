# imports for the exrcise - part 1
# you can add more if you wish (but it is not really needed)
import numpy as np
import matplotlib.pyplot as plt

# pytorch imports
import torch
from torch.utils.data import DataLoader, Dataset, ConcatDataset
import torchvision

from CelebA_VIAE.CelebA_VIAE_sample import Z_E_DIM
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
###########################################################################################
'Train Load!'

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

# male_subset = celeba_train_dataset[male_indices[0].tolist()]
# female_subset = celeba_train_dataset[female_indices[0]]

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
"Test Load"

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

celeba_dataset = torchvision.datasets.ImageFolder(root=data_root, transform=transform)

# Gender attribute is at index 20: 1 = Male, -1 = Female
import pandas as pd
list_eval_partition = pd.read_csv("./data/celeba/list_eval_partition.csv")
test_indices = torch.where(torch.tensor(list_eval_partition['partition'] == 2))#[0]male_subset = torch.utils.data.Subset(celeba_test_dataset, male_indices[0].tolist())
celeba_test_dataset = torch.utils.data.Subset(celeba_dataset, test_indices[0].tolist())

all_attributes = pd.read_csv("./data/celeba/list_attr_celeba.csv")
all_attributes_test = all_attributes.iloc[test_indices[0].tolist()]

# male_indices = torch.where(torch.tensor(all_attributes['Male'] == 1))#[0]
# female_indices = torch.where(torch.tensor(all_attributes['Male'] == -1))#[0]

male_indices = torch.where(
    (torch.tensor(all_attributes['Male'] == 1)) &
    (torch.tensor(list_eval_partition['partition'] == 2))
)
female_indices = torch.where(
    (torch.tensor(all_attributes['Male'] == -1)) &
    (torch.tensor(list_eval_partition['partition'] == 2))
)

male_subset = torch.utils.data.Subset(celeba_test_dataset, male_indices[0].tolist())
female_subset = torch.utils.data.Subset(celeba_test_dataset, female_indices[0].tolist())

# male_subset = celeba_test_dataset[male_indices[0].tolist()]
# female_subset = celeba_test_dataset[female_indices[0]]


# Optional: Wrap in a DataLoader
celeba_x_test_e1 = male_subset #DataLoader(male_subset, batch_size=batch_size_test, shuffle=True)
celeba_x_test_e2 = female_subset#DataLoader(female_subset, batch_size=batch_size_test, shuffle=True)


# celeba_test_e1 = celeba_x_test_e1.dataset[male_indices[0]]
#
# celeba_test_e2 = celeba_x_test_e2.dataset[female_indices[0]]
# celeba_test_e1 = celeba_test_dataset[male_indices[0].tolist()[0]]
# celeba_test_e2 = [celeba_test_dataset[i] for i in female_indices[0].tolist()]
from torch.utils.data import DataLoader, Subset

# Set batch size for faster loading

# Create a DataLoader for your subsetssbssss
subs1 = Subset(celeba_test_dataset, male_indices[0].tolist())
subs2 = Subset(celeba_test_dataset, female_indices[0].tolist())

celeba_x_test_e1 = DataLoader(male_subset, batch_size=batch_size_test, shuffle=True)
celeba_x_test_e2 = DataLoader(female_subset, batch_size=batch_size_test, shuffle=True)
###############################################################################

"Parameters"

HIDDEN_SIZE=256
X_DIM=64*64*3
Z_C_DIM=56
Z_E_DIM = 8
if torch.cuda.is_available():
    torch.cuda.current_device()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
vae = Vae_Irm(z_c_dim=Z_C_DIM, z_e_dim=Z_E_DIM, device=device).to(device)

###########################################################################################################
"Examples"

fig, axes = plt.subplots(1, 2, figsize=(15, 5))  # Create a figure with 1 row and 2 columns
# Plot for e=1
# to_plot_e1 = torch.cat((train_loader_e1.dataset.data[6,:,:].permute(1, 2, 0), zero_channel), dim=2)
to_plot_e1 = celeba_x_train_e1.dataset[6][0].permute(1, 2, 0)
axes[0].imshow(to_plot_e1)
axes[0].axes.get_xaxis().set_ticks([])
axes[0].axes.get_yaxis().set_ticks([])
axes[0].set_title("e=1")  # Add label for e=1
#
# Plot for e=2
# to_plot_e2 = torch.cat((train_loader_e2.dataset.data[6,:,:].permute(1, 2, 0), zero_channel), dim=2)
to_plot_e2 = celeba_x_train_e2.dataset[6][0].permute(1, 2, 0)
axes[1].imshow(to_plot_e2)
axes[1].axes.get_xaxis().set_ticks([])
axes[1].axes.get_yaxis().set_ticks([])
axes[1].set_title("e=2")  # Add label for e=2
plt.tight_layout()  # Adjust layout to avoid overlap
plt.show()
# fig.savefig('Testenv_combined.png')  # Save the combined figure

###################################################################################################

'Fine Tune!- Turned off'

vae.load_state_dict(torch.load('vae_irm.pth'))
vae.requires_grad_(True)
# vae.encoder_env2.load_state_dict(vae.encoder_env1.state_dict())
vae.encoder_causal.requires_grad_(False)
vae.decoder.requires_grad_(False)

# vae,train_recon_errors,train_kls,train_losses= train_beta_vae(1, train_loader_e1, train_loader_e2, NUM_EPOCHS=5, vae=vae, fine_tune=1)
######################################################################################################
"Inference"
test_x_e1 = [celeba_x_train_e1.dataset[i][0] for i in range(len(celeba_x_train_e1.dataset))]
test_y_e1= [celeba_x_train_e1.dataset[i][1] for i in range(len(celeba_x_train_e1.dataset))]
test_e1 = enumerate(celeba_x_train_e1.dataset)
batch_idx_e1, (test_data_e1, test_targets_e1) = next(test_e1)

# test_data_e1.shape
#
# torch.Size([1000, 1, 28, 28])

test_x_e2 = [celeba_x_train_e2.dataset[i][0] for i in range(len(celeba_x_train_e2.dataset))]
test_y_e2= [celeba_x_train_e2.dataset[i][1] for i in range(len(celeba_x_train_e2.dataset))]
test_e2 = enumerate(celeba_x_train_e2.dataset)
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
for i in range(6):
    plt.subplot(2,3,i+1)
    plt.tight_layout()
    x_e = test_x_e1[i+1].to(device).float()#/255
    to_plot = x_e.cpu().detach().permute(1, 2, 0)
    plt.imshow(to_plot, cmap='gray')
    plt.title("e source: 1")
    plt.xticks([])
    plt.yticks([])
# for i in range(3):
#     plt.subplot(2,3,i+4)
#     plt.tight_layout()
#     x_e = test_x_e2[i+0].to(device).float()#/255
#     to_plot = x_e.cpu().detach().permute(1, 2, 0)
#     plt.imshow(to_plot, cmap='gray')
#     plt.title("Label: {}, e source: 2".format(test_y_e2[i+0]))
#     plt.xticks([])
#     plt.yticks([])
fig
plt.show()
#####################################################################
"Inf!"

fig = plt.figure(figsize=(9, 5))
for i in range(6):
    plt.subplot(2,3,i+1)
    plt.tight_layout()
    # x_e = test_x_e2[i+15].view(-1, X_DIM).to(device).float()/255
    x_e = test_x_e1[i+1].to(device).unsqueeze(0).float()#/255
    z, mu, logvar = vae.encode(x_e,1)
    # z_e = torch.randn(1, 8)
    z_e = z[:,56:64]
    z_e[:,7] = z_e[:,7] + 100
    z_e[:, 6] = z_e[:, 6] - 100
    z[:,56:64] = z_e.to(device)
    # z_e = torch.randn(1, 32)
    # z_e[0,30] = z_e[0,30]+100
    ##
    # z_e12 = torch.randn(1, 10)
    # z_e12[0, 8] = z_e12[0, 8] + 50
    # z_e12[0, 9] = z_e12[0, 9] + 50
    # z_c, mu_c, logvar_c = vae.encoder_causal(x_e.to(device), z_e12.to(device))
    # z = torch.cat((z_c, z_e.to(device)), dim=1)
    ##
    # z2, mu2, logvar2 = vae.encode(x_e, 2)
    # z2[0, 32:64] = z_e.to(device)
    # z = (z + z2) / 2
    ##
    x = vae.decode(z)
    # plt.imshow(x.cpu().view(28,28).detach(), cmap='gray', interpolation='none')
    # to_plot = torch.cat((x_e.cpu().detach().squeeze().permute(1, 2, 0), zero_channel), dim=2)
    to_plot = x.cpu().detach().squeeze().permute(1, 2, 0)
    plt.imshow(to_plot, cmap='gray')
    plt.title("e source: 1")
    plt.xticks([])
    plt.yticks([])
# for i in range(3):
#     plt.subplot(2,3,i+4)
#     plt.tight_layout()
#     # x_e = test_x_e2[i+15].view(-1, X_DIM).to(device).float()/255
#     x_e = test_x_e2[i+0].to(device).unsqueeze(0).float()#/255
#     z, mu, logvar = vae.encode(x_e,2)
#     z_e = torch.randn(1, 25)
#     z_e[0,24] = z_e[0,24]+100
#     z[0,25:50] = z_e.to(device)
#     # z_e = torch.randn(1, 10)
#     # z_e[0,8] = z_e[0,8]+100
#     # z_c, mu_c, logvar_c = vae.encoder_causal(x_e.to(device), z_e.to(device))
#     # z = torch.cat((z_c, z_e.to(device)), dim=1)
#     ##
#     # z2, mu2, logvar2 = vae.encode(x_e, 2)
#     # z2[0, 32:64] = z_e.to(device)
#     # z = (z + z2) / 2
#     ##
#     x = vae.decode(z)
#     # plt.imshow(x.cpu().view(28,28).detach(), cmap='gray', interpolation='none')
#     # to_plot = torch.cat((x_e.cpu().detach().squeeze().permute(1, 2, 0), zero_channel), dim=2)
#     to_plot = x.cpu().detach().squeeze().permute(1, 2, 0)
#     plt.imshow(to_plot, cmap='gray')
#     plt.title("Label: {}, e source: 2".format(test_y_e2[i+0]))
#     plt.xticks([])
#     plt.yticks([])
fig
plt.show()