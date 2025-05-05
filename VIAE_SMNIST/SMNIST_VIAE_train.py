import numpy as np
import matplotlib.pyplot as plt

# pytorch imports
import torch
from torch.utils.data import DataLoader, Dataset, ConcatDataset
import torchvision
from SMNIST_VIAE_model import train_beta_vae, Vae_Irm, VaeEncoderEnv, VaeEncoderCausal, VaeDecoder

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

# random_seed = 1
# torch.backends.cudnn.enabled = False
# torch.manual_seed(random_seed)

###########################################################################################
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
"Disturbances"
#
for i in range(len(temp_train_x_e1)):
    changePa = np.random.rand(1)
    if (changePa < 1):
        'SMNIST'
        temp_train_x_e1[i, 0:7, 0:7] = 255
        'RMNIST'
        # temp_train_x_e1[i, :, :] = torchvision.transforms.functional.rotate(temp_train_x_e1[i, :, :].unsqueeze(0).unsqueeze(0), angle=45)


for i in range(len(temp_train_x_e2)):
    # changePa = np.random.rand(1)
    if (changePa < 1):
        'SMNIST'
        temp_train_x_e2[i, -7:-1, -7:-1] = 255
        'RMNIST'
        # temp_train_x_e2[i, :, :] = torchvision.transforms.functional.rotate(temp_train_x_e2[i, :, :].unsqueeze(0).unsqueeze(0), angle=315)


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
#############################################################################

fig, axes = plt.subplots(1, 2, figsize=(15, 5))  # Create a figure with 1 row and 2 columns
# Plot for e=1
# to_plot_e1 = torch.cat((train_loader_e1.dataset.data[6,:,:].permute(1, 2, 0), zero_channel), dim=2)
to_plot_e1 = train_loader_e1.dataset.data[6, :, :].squeeze()
axes[0].imshow(to_plot_e1, cmap='gray')
axes[0].axes.get_xaxis().set_ticks([])
axes[0].axes.get_yaxis().set_ticks([])
axes[0].set_title("e=1")  # Add label for e=1
#
# Plot for e=2
# to_plot_e2 = torch.cat((train_loader_e2.dataset.data[6,:,:].permute(1, 2, 0), zero_channel), dim=2)
to_plot_e2 = train_loader_e2.dataset.data[6, :, :].squeeze()
axes[1].imshow(to_plot_e2, cmap='gray')
axes[1].axes.get_xaxis().set_ticks([])
axes[1].axes.get_yaxis().set_ticks([])
axes[1].set_title("e=2")  # Add label for e=2
plt.tight_layout()  # Adjust layout to avoid overlap
plt.show()
fig.savefig('Trainenv_combined.png')  # Save the combined figure
####################################################################################################
"Training!"
vae,train_recon_errors,train_kls,train_losses= train_beta_vae(1, train_loader_e1, train_loader_e2, NUM_EPOCHS=100)

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

# Save the model's parameters
torch.save(vae.state_dict(), 'vae_irm.pth')