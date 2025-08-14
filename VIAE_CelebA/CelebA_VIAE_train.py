import numpy as np
import matplotlib.pyplot as plt

# pytorch imports
import torch
from torch.utils.data import DataLoader, Dataset, ConcatDataset
import torchvision
from CelebA_VIAE_model import train_beta_vae, Vae_Irm, VaeEncoderEnv1, VaeEncoderEnv2, VaeEncoderCausal, VaeDecoder

datapath = '/files/'

n_epochs = 50
batch_size_train = 64
batch_size_test = 128
learning_rate = 0.002#0.00005
momentum = 0.5
log_interval = 10
lam = 50
w_dis_history = np.array([])
acc_vec = np.array([])
num_of_channels = 3

"Deterministic Settings"
# random_seed = 1
# torch.backends.cudnn.enabled = False
# torch.manual_seed(random_seed)

###########################################################################################
'Train Load!'

import torchvision.transforms as transforms
# Set where to store the data
data_root = "../data/"

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
# male_indices = torch.where(torch.tensor(all_attributes['Male'] == 1))#[0]
# female_indices = torch.where(torch.tensor(all_attributes['Male'] == -1))#[0]

list_eval_partition = pd.read_csv("./data/celeba/list_eval_partition.csv")
# train_indices = torch.where(torch.tensor(list_eval_partition['partition'] == 0))#[0]male_subset = torch.utils.data.Subset(celeba_test_dataset, male_indices[0].tolist())
# celeba_test_dataset = torch.utils.data.Suset(celeba_dataset, test_indices[0].tolist())

male_indices = torch.where(
    (torch.tensor(all_attributes['Male'] == 1)) &
    (torch.tensor(list_eval_partition['partition'] == 0))
)
female_indices = torch.where(
    (torch.tensor(all_attributes['Male'] == -1)) &
    (torch.tensor(list_eval_partition['partition'] == 0))
)

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
# celeba_train_e1 = celeba_train_dataset[male_indices[0].tolist()[0]]
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
"Examples"

fig, axes = plt.subplots(1, 2, figsize=(15, 5))  # Create a figure with 1 row and 2 columns
# Plot for e=1
to_plot_e1 = celeba_x_train_e1.dataset[6][0].permute(1, 2, 0)
axes[0].imshow(to_plot_e1)
axes[0].axes.get_xaxis().set_ticks([])
axes[0].axes.get_yaxis().set_ticks([])
axes[0].set_title("e=1")  # Add label for e=1
#
# Plot for e=2

to_plot_e2 = celeba_x_train_e2.dataset[6][0].permute(1, 2, 0)
axes[1].imshow(to_plot_e2)
axes[1].axes.get_xaxis().set_ticks([])
axes[1].axes.get_yaxis().set_ticks([])
axes[1].set_title("e=2")  # Add label for e=2
plt.tight_layout()  # Adjust layout to avoid overlap
plt.show()
fig.savefig('Trainenv_combined.png')  # Save the combined figure

"More"
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
    ax.imshow(to_plot, cmap='gray')
    ax.axes.get_xaxis().set_ticks([])
    ax.axes.get_yaxis().set_ticks([])
    if j==0:
        ax.set_title("Environment 2")

plt.show()


# ####################################################################################################
"Training!"
vae,train_recon_errors,train_kls,train_losses= train_beta_vae(0.00025, celeba_x_train_e1, celeba_x_train_e2, NUM_EPOCHS=100)
#0.00025
# Save the model's parameters
torch.save(vae.state_dict(), 'vae_irm.pth')

fig = plt.figure(figsize=(10,4))
ax1 = fig.add_subplot(1,2,1)
ax1.set_title("Reconstruction error vs epochs")
ax1.set_xlabel("Epoch")
ax1.set_ylabel("Reconstruction error")
ax2 = fig.add_subplot(1,2,2)
ax2.set_title("KL-divergence vs epochs")
ax2.set_xlabel("Epoch")
ax2.set_ylabel("KL-divergence")
legend=[]
ax1.plot(train_recon_errors)
ax2.plot(train_kls)
ax1.legend(legend)
ax2.legend(legend)

plt.show()
# fig.savefig('Training_Metrics.png')
