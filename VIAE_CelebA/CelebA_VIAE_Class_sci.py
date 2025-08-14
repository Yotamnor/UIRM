import numpy as np

import matplotlib.pyplot as plt

# pytorch imports
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, ConcatDataset
import torchvision
import sklearn
from sklearn import linear_model
from sklearn.metrics import accuracy_score
# from CMNIST_VIAE.CMNIST_VIAE_model import train_beta_vae, Vae_Irm, VaeEncoderEnv1, VaeEncoderEnv2, VaeEncoderCausal, VaeDecoder
from CelebA_VIAE_model import train_beta_vae, Vae_Irm, VaeEncoderEnv1, VaeEncoderEnv2, VaeEncoderCausal, VaeDecoder

#######################################################################################################

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
# target_attr = 'Oval_Face'
# ##########################################################################################################
# 'Train Load!'
#
# from torchvision.datasets import CelebA
# import torchvision.transforms as transforms
#
# # Set where to store the data
# data_root = "/home/yotamnor/files/Yotam_env/VAE_VAE/CelebA_VIAE/data" #"./data/"#
#
# # Define any transforms (resize, normalize, etc.)
# transform = transforms.Compose([
#     transforms.Resize(64),
#     transforms.CenterCrop(64),
#     transforms.ToTensor()
# ])
#
# # celeba_train_dataset = torchvision.datasets.ImageFolder(root=data_root, transform=transform)
# celeba_train_dataset = CelebA(root=data_root, split='train', target_type='attr', transform=transform, download=False)
#
#
# # Gender attribute is at index 20: 1 = Male, -1 = Female
# import pandas as pd
# all_attributes = pd.read_csv("./data/celeba/list_attr_celeba.csv")
#
# list_eval_partition = pd.read_csv("./data/celeba/list_eval_partition.csv")
#
# male_indices = torch.where(
#     (torch.tensor(all_attributes['Male'] == 1)) &
#     (torch.tensor(list_eval_partition['partition'] == 0))
# )
# female_indices = torch.where(
#     (torch.tensor(all_attributes['Male'] == -1)) &
#     (torch.tensor(list_eval_partition['partition'] == 0))
# )
#
# male_subset = torch.utils.data.Subset(celeba_train_dataset, male_indices[0].tolist())
# female_subset = torch.utils.data.Subset(celeba_train_dataset, female_indices[0].tolist())
#
#
# celeba_train_e1 = celeba_train_dataset[male_indices[0].tolist()[0]]
# from torch.utils.data import DataLoader, Subset
#
# # batch_size = 512  # Adjust based on your available memory
#
# # Create a DataLoader for your subsetssbssss
# subs1 = Subset(celeba_train_dataset, male_indices[0].tolist())
# subs2 = Subset(celeba_train_dataset, female_indices[0].tolist())
#
# celeba_x_train_e1 = DataLoader(subs1, batch_size=batch_size_train, shuffle=False)
# celeba_x_train_e2 = DataLoader(subs2, batch_size=batch_size_train, shuffle=False)
# ###############################################################################################################################################################################################################
# 'Test Load!'
#
# from torchvision.datasets import CelebA
# import torchvision.transforms as transforms
#
# # Set where to store the data
# data_root = "/home/yotamnor/files/Yotam_env/VAE_VAE/CelebA_VIAE/data" #"./data/"#
#
# # Define any transforms (resize, normalize, etc.)
# transform = transforms.Compose([
#     transforms.Resize(64),
#     transforms.CenterCrop(64),
#     transforms.ToTensor()
# ])
#
# # celeba_test_dataset = torchvision.datasets.ImageFolder(root=data_root, transform=transform)
# celeba_test_dataset = CelebA(root=data_root, split='test', target_type='attr', transform=transform, download=False)
#
# # Gender attribute is at index 20: 1 = Male, -1 = Female
# import pandas as pd
# all_attributes = pd.read_csv("./data/celeba/list_attr_celeba.csv")
#
# list_eval_partition = pd.read_csv("./data/celeba/list_eval_partition.csv")
#
# male_indices = torch.where(
#     (torch.tensor(all_attributes['Male'] == 1)) &
#     (torch.tensor(list_eval_partition['partition'] == 2))
# )
# female_indices = torch.where(
#     (torch.tensor(all_attributes['Male'] == -1)) &
#     (torch.tensor(list_eval_partition['partition'] == 2))
# )
#
# male_subset = torch.utils.data.Subset(celeba_test_dataset, male_indices[0].tolist())
# female_subset = torch.utils.data.Subset(celeba_test_dataset, female_indices[0].tolist())
#
# from torch.utils.data import DataLoader, Subset
#
# # batch_size = 512  # Adjust based on your available memory
#
# # Create a DataLoader for your subsetssbssss
# subs1 = Subset(celeba_test_dataset, male_indices[0].tolist())
# subs2 = Subset(celeba_test_dataset, female_indices[0].tolist())
#
# celeba_x_test_e1 = DataLoader(subs1, batch_size=batch_size_test, shuffle=False)
# celeba_x_test_e2 = DataLoader(subs2, batch_size=batch_size_test, shuffle=False)
#
# #######################################################################################
# "Labels Indices"
# male_train_ind = np.where(
#     (torch.tensor(all_attributes['Male'] == 1)) &
#     # (torch.tensor(all_attributes['Mouth_Slightly_Open'] == 1)) &
#     (torch.tensor(list_eval_partition['partition'] == 0))
# )
# male_test_ind = np.where(
#     (torch.tensor(all_attributes['Male'] == 1)) &
#     # (torch.tensor(all_attributes['Mouth_Slightly_Open'] == 1)) &
#     (torch.tensor(list_eval_partition['partition'] == 2))
# )
# # mso_train_ind = np.where(
# #     # (torch.tensor(all_attributes['Male'] == 1)) &
# #     (torch.tensor(all_attributes['Mouth_Slightly_Open'] == 1)) &
# #     (torch.tensor(list_eval_partition['partition'] == 0))
# # )
# # mso_test_ind = np.where(
# #     # (torch.tensor(all_attributes['Male'] == 1)) &
# #     (torch.tensor(all_attributes['Mouth_Slightly_Open'] == 1)) &
# #     (torch.tensor(list_eval_partition['partition'] == 2))
# # )
# female_train_ind = np.where(
#     (torch.tensor(all_attributes['Male'] == -1)) &
#     # (torch.tensor(all_attributes['Mouth_Slightly_Open'] == 1)) &
#     (torch.tensor(list_eval_partition['partition'] == 0))
# )
# female_test_ind = np.where(
#     (torch.tensor(all_attributes['Male'] == -1)) &
#     # (torch.tensor(all_attributes['Mouth_Slightly_Open'] == 1)) &
#     (torch.tensor(list_eval_partition['partition'] == 2))
# )
# # mc_train_ind = np.where(
# #     # (torch.tensor(all_attributes['Male'] == -1)) &
# #     (torch.tensor(all_attributes['Mouth_Slightly_Open'] == -1)) &
# #     (torch.tensor(list_eval_partition['partition'] == 0))
# # )
# # mc_test_ind = np.where(
# #     # (torch.tensor(all_attributes['Male'] == -1)) &
# #     (torch.tensor(all_attributes['Mouth_Slightly_Open'] == -1)) &
# #     (torch.tensor(list_eval_partition['partition'] == 2))
# # )
#
# ####################################################################################################
# "Data Arrange"
#
# train_loader_e1 = celeba_x_train_e1
# train_loader_e2 = celeba_x_train_e2
# test_loader_e1 = celeba_x_test_e1
# test_loader_e2 = celeba_x_test_e2
#
# "Data"
# train_dataset_x_e1= train_loader_e1.dataset
# train_dataset_x_e2= train_loader_e2.dataset
# test_dataset_x_e1= test_loader_e1.dataset
# test_dataset_x_e2= test_loader_e2.dataset
#
# "Labels"
# train_y_e1= torch.from_numpy(all_attributes['Oval_Face'].iloc[male_train_ind].values).long()
# train_y_e2= torch.from_numpy(all_attributes['Oval_Face'].iloc[female_train_ind].values).long()
# test_y_e1= torch.from_numpy(all_attributes['Oval_Face'].iloc[male_test_ind].values).long()
# test_y_e2= torch.from_numpy(all_attributes['Oval_Face'].iloc[female_test_ind].values).long()
##############################################################################################################################
#####################################################################################################
import torch
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import CelebA
import torchvision.transforms as transforms
import numpy as np

# Settings
data_root = "/home/yotamnor/files/Yotam_env/VAE_VAE/CelebA_VIAE/data"
batch_size_train = 64
batch_size_test = 128

# Transform
transform = transforms.Compose([
    transforms.Resize(64),
    transforms.CenterCrop(64),
    transforms.ToTensor()
])

# Attribute you want to predict
target_attr = 'Mouth_Slightly_Open'#'Bangs'#Eyeglasses

# Load full dataset with all attributes
celeba_train = CelebA(root=data_root, split='train', target_type='attr', transform=transform, download=True)
celeba_test = CelebA(root=data_root, split='test', target_type='attr', transform=transform, download=True)

# Convert labels from -1/+1 to 0/1
celeba_train.attr = (celeba_train.attr + 1) // 2
celeba_test.attr = (celeba_test.attr + 1) // 2

# Indices for each gender in train/test
male_train_idx = (celeba_train.attr[:, 20] == 1).nonzero(as_tuple=True)[0]
female_train_idx = (celeba_train.attr[:, 20] == 0).nonzero(as_tuple=True)[0]
male_test_idx = (celeba_test.attr[:, 20] == 1).nonzero(as_tuple=True)[0]
female_test_idx = (celeba_test.attr[:, 20] == 0).nonzero(as_tuple=True)[0]

# Create environment-specific subsets
train_dataset_e1 = Subset(celeba_train, male_train_idx.tolist())
train_dataset_e2 = Subset(celeba_train, female_train_idx.tolist())
test_dataset_e1 = Subset(celeba_test, male_test_idx.tolist())
test_dataset_e2 = Subset(celeba_test, female_test_idx.tolist())

# Dataloaders
train_loader_e1 = DataLoader(train_dataset_e1, batch_size=batch_size_train, shuffle=False)
train_loader_e2 = DataLoader(train_dataset_e2, batch_size=batch_size_train, shuffle=False)
test_loader_e1 = DataLoader(test_dataset_e1, batch_size=batch_size_test, shuffle=False)
test_loader_e2 = DataLoader(test_dataset_e2, batch_size=batch_size_test, shuffle=False)

# Extract labels (Oval_Face) for each subset
train_y_e1 = celeba_train.attr[male_train_idx, celeba_train.attr_names.index(target_attr)]
train_y_e2 = celeba_train.attr[female_train_idx, celeba_train.attr_names.index(target_attr)]
test_y_e1 = celeba_test.attr[male_test_idx, celeba_test.attr_names.index(target_attr)]
test_y_e2 = celeba_test.attr[female_test_idx, celeba_test.attr_names.index(target_attr)]

#####################################################################################################################
"NN Init"

HIDDEN_SIZE=256
X_DIM= 64*64*3
Z_C_DIM= 56
Z_E_DIM= 8

if torch.cuda.is_available():
    torch.cuda.current_device()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
vae = Vae_Irm(z_c_dim=Z_C_DIM, z_e_dim=Z_E_DIM, device=device).to(device)
vae.load_state_dict(torch.load('vae_irm.pth'))#../CMNIST_VIAE/
vae.requires_grad_(False)
vae.eval()

#################################################################################################################
"Get Embeddings Training!"

flag=0
for batch_e1 in train_loader_e1:
    z_1_temp, mu_1_temp, logvar_1_temp = vae.encode(batch_e1[0].to(device),1)
    z_1_detach = mu_1_temp.detach().cpu().squeeze()

    if flag == 0 :
        z_1 = z_1_detach
        flag=1
    else:
        z_1 = torch.cat((z_1, z_1_detach), 0)
flag=0
for batch_e2 in train_loader_e2:
    z_2_temp, mu_2_temp, logvar_2_temp = vae.encode(batch_e2[0].to(device),2)
    z_2_detach = mu_2_temp.detach().cpu().squeeze()

    if flag == 0 :
        z_2 = z_2_detach
        flag=1
    else:
        z_2 = torch.cat((z_2, z_2_detach), 0)
#
# z_1, mu_1, logvar_1 = vae.encode(temp_train_dataset_e1.data.to(device),1)
# z_2, mu_2, logvar_2 = vae.encode(temp_train_dataset_e2.data.to(device),2)

# z_1_detach = mu_1.detach().cpu()
# z_2_detach = mu_2.detach().cpu()
# z_1 = z_1_detach
# z_2 = z_2_detach

z_1_I = z_1[:,0:Z_C_DIM]#.cpu()
z_1_e = z_1[:,Z_C_DIM: Z_C_DIM + Z_E_DIM]#.cpu()
z_2_I = z_2[:,0:Z_C_DIM]#.cpu()
z_2_e = z_2[:,Z_C_DIM:Z_C_DIM + Z_E_DIM]#.cpu()

z_I_train = torch.cat((z_1_I, z_2_I), 0)
z_e_train = torch.cat((z_1_e, z_2_e), 0)

labels_train = torch.cat((train_y_e1, train_y_e2), 0)

e1_gt = train_y_e1*0
e1_gt[:] = 1
e1_gt = e1_gt.long()
e2_gt = train_y_e2*0
e2_gt[:] = -1
e2_gt = e2_gt.long()
env_train = torch.cat((e1_gt, e2_gt), 0)
#################################################################################################################
"Get Embeddings Test!"
#
flag=0
for batch_e1 in test_loader_e1:
    z_1_temp, mu_1_temp, logvar_1_temp = vae.encode(batch_e1[0].to(device),1)
    z_1_detach = mu_1_temp.detach().cpu().squeeze()

    if flag == 0:
        z_1 = z_1_detach
        flag=1
    else:
        z_1 = torch.cat((z_1, z_1_detach), 0)
flag=0
for batch_e2 in test_loader_e2:
    z_2_temp, mu_2_temp, logvar_2_temp = vae.encode(batch_e2[0].to(device),2)
    z_2_detach = mu_2_temp.detach().cpu().squeeze()

    if flag == 0:
        z_2 = z_2_detach
        flag=1
    else:
        z_2 = torch.cat((z_2, z_2_detach), 0)

#
# z_1, mu_1, logvar_1 = vae.encode(temp_test_dataset_e1.data.to(device),1)
# z_2, mu_2, logvar_2 = vae.encode(temp_test_dataset_e2.data.to(device),2)
#
# z_1_detach = mu_1.detach().cpu()
# z_2_detach = mu_2.detach().cpu()
# z_1 = z_1_detach
# z_2 = z_2_detach
#
# # z_1_detach = mu_1.detach()
# # z_2_detach = mu_2.detach()

z_1_I = z_1[:,0:Z_C_DIM]#.cpu()
z_1_e = z_1[:,Z_C_DIM: Z_C_DIM + Z_E_DIM]#.cpu()
z_2_I = z_2[:,0:Z_C_DIM]#.cpu()
z_2_e = z_2[:,Z_C_DIM:Z_C_DIM + Z_E_DIM]#.cpu()


z_I_test = torch.cat((z_1_I, z_2_I), 0)
z_e_test = torch.cat((z_1_e, z_2_e), 0)

labels_test = torch.cat((test_y_e1, test_y_e2), 0)

e1_gt = test_y_e1*0
e1_gt[:] = 1
e1_gt = e1_gt.long()
e2_gt = test_y_e2*0
e2_gt[:] = -1
e2_gt = e2_gt.long()
env_test = torch.cat((e1_gt, e2_gt), 0)
##############################################################################################################
# "Baseline Training"
#
# # X_DIM = 64*64
#
# train_x_e1 = torch.stack([train_dataset_x_e1[i][0] for i in range(len(train_dataset_x_e1))])
# train_x_e2 = torch.stack([train_dataset_x_e2[i][0] for i in range(len(train_dataset_x_e2))])
# test_x_e1 = torch.stack([test_dataset_x_e1[i][0] for i in range(len(test_dataset_x_e1))])
# test_x_e2 = torch.stack([test_dataset_x_e2[i][0] for i in range(len(test_dataset_x_e2))])
#
# X_train =  torch.cat((train_x_e1, train_x_e2), 0).squeeze().view(-1, X_DIM)
# X_test = torch.cat((test_x_e1, test_x_e2), 0).squeeze().view(-1, X_DIM)
#
# W_B2L = linear_model.LogisticRegression()
# W_B2E = linear_model.LogisticRegression()
#
# W_B2L.fit(X_train, labels_train)
# W_B2E.fit(X_train, env_train)

##############################################################################################################
"Training"

W_I2L = linear_model.LogisticRegression(class_weight='balanced')
W_I2E = linear_model.LogisticRegression(class_weight='balanced')
W_E2L = linear_model.LogisticRegression(class_weight='balanced')
W_E2E = linear_model.LogisticRegression(class_weight='balanced')

W_I2L.fit(z_I_train, labels_train)
W_I2E.fit(z_I_train, env_train)
W_E2L.fit(z_e_train, labels_train)
W_E2E.fit(z_e_train, env_train)

###################################################################################################################################################
'Test Loss and Classification Accuracy'
#
# W_B2L_pred = W_B2L.predict(X_train)
# W_B2E_pred = W_B2E.predict(X_train)
#
# W_I2L_pred = W_I2L.predict(z_I_train)
# W_I2E_pred = W_I2E.predict(z_I_train)
# W_E2L_pred = W_E2L.predict(z_e_train)
# W_E2E_pred = W_E2E.predict(z_e_train)
#
# # print("B2L Train Accuracy:", accuracy_score(labels_train, W_B2L_pred))
# # print("B2E Train Accuracy:", accuracy_score(env_train, W_B2E_pred))
#
# print("I2L Train Accuracy:", accuracy_score(labels_train, W_I2L_pred))
# print("I2E Train Accuracy:", accuracy_score(env_train, W_I2E_pred))
# print("E2L Train Accuracy:", accuracy_score(labels_train, W_E2L_pred))
# print("E2E Train Accuracy:", accuracy_score(env_train, W_E2E_pred))

# W_B2L_pred = W_B2L.predict(X_test)
# W_B2E_pred = W_B2E.predict(X_test)

W_I2L_pred = W_I2L.predict(z_I_test)
W_I2E_pred = W_I2E.predict(z_I_test)
W_E2L_pred = W_E2L.predict(z_e_test)
W_E2E_pred = W_E2E.predict(z_e_test)

# print("B2L Test Accuracy:", accuracy_score(labels_test, W_B2L_pred))
# print("B2E Test Accuracy:", accuracy_score(env_test, W_B2E_pred))

print("I2L Test Accuracy:", accuracy_score(labels_test, W_I2L_pred))
print("I2E Test Accuracy:", accuracy_score(env_test, W_I2E_pred))
print("E2L Test Accuracy:", accuracy_score(labels_test, W_E2L_pred))
print("E2E Test Accuracy:", accuracy_score(env_test, W_E2E_pred))