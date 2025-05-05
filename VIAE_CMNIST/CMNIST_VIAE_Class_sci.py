import numpy as np

# pytorch imports
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, ConcatDataset
import torchvision
from sklearn import linear_model
from sklearn.metrics import accuracy_score
from CMNIST_VIAE_model import train_beta_vae, Vae_Irm, VaeEncoderEnv, VaeEncoderCausal, VaeDecoder
##########################################################################################################
"Neural Nets"
class Label_Class(torch.nn.Module):

    def __init__(self, device=torch.device("cpu")):
        super(Label_Class, self).__init__()

        self.lin = nn.Linear(10, 10, bias = True)

    def forward(self, x):

        y_hat = self.lin(x)
        return y_hat

class Env_Class(torch.nn.Module):

    def __init__(self, device=torch.device("cpu")):
        super(Env_Class, self).__init__()

        self.lin = nn.Linear(10, 2, bias = True)

    def forward(self, x):

        e_hat = self.lin(x)
        return e_hat
###############################################################################################################
"Hyper Parameters"

n_epochs = 20
batch_size_train = 128#64
batch_size_test = 128
learning_rate = 1e-3
momentum = 0.5
acc_vec = np.array([])
log_interval = 10

###################################################################################################################
"Data Load"

transform = torchvision.transforms.ToTensor()

test_loader_e1 = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST('../datasets/', train=False, download=True,
                               transform=transform, target_transform=None), batch_size=batch_size_test, shuffle=True)

test_loader_e2 = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST('../datasets/', train=False, download=True,
                               transform=transform, target_transform=None), batch_size=batch_size_test, shuffle=True)

train_loader_e1 = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST('../datasets/', train=True, download=True,
                               transform=transform, target_transform=None), batch_size=batch_size_train, shuffle=True)
train_loader_e2 = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST('../datasets/', train=True, download=True,
                               transform=transform, target_transform=None), batch_size=batch_size_train, shuffle=True)

"Data"
train_x_e1= train_loader_e1.dataset.data
train_x_e2= train_loader_e2.dataset.data
test_x_e1= test_loader_e1.dataset.data
test_x_e2= test_loader_e2.dataset.data
"Labels"
train_y_e1= train_loader_e1.dataset.targets
train_y_e2= train_loader_e2.dataset.targets
test_y_e1= test_loader_e1.dataset.targets
test_y_e2= test_loader_e2.dataset.targets

"Env Data"
temp_train_x_e1 = train_x_e1[0:int(train_y_e1.shape[0]/2)]
temp_train_x_e2 = train_x_e2[int(train_y_e1.shape[0]/2):train_y_e1.shape[0]]
temp_test_x_e1 = test_x_e1[0:int(test_y_e1.shape[0]/2)]
temp_test_x_e2 = test_x_e2[int(test_y_e1.shape[0]/2):test_y_e1.shape[0]]
"Env Labels"
temp_train_y_e1= train_y_e1[0:int(train_y_e1.shape[0]/2)]
temp_train_y_e2= train_y_e2[int(train_y_e1.shape[0]/2):train_y_e1.shape[0]]
temp_test_y_e1= test_y_e1[0:int(test_y_e1.shape[0]/2)]
temp_test_y_e2= test_y_e2[int(test_y_e1.shape[0]/2):test_y_e1.shape[0]]
###############################################################################################################
"Disturbances"

zero_channel_train_data = torch.zeros((30000,1,28,28))
zero_channel_test_data = torch.zeros((5000,1,28,28))

temp_train_x_e1 = torch.cat((temp_train_x_e1.unsqueeze(1), zero_channel_train_data, zero_channel_train_data), dim=1)
temp_train_x_e2 = torch.cat((zero_channel_train_data, temp_train_x_e2.unsqueeze(1), zero_channel_train_data), dim=1)

temp_test_x_e1 = torch.cat((temp_test_x_e1.unsqueeze(1), zero_channel_test_data, zero_channel_test_data), dim=1)
temp_test_x_e2 = torch.cat((zero_channel_test_data, temp_test_x_e2.unsqueeze(1), zero_channel_test_data), dim=1)

#####################################################################################################################
"Putback"

temp_train_dataset_e1 = torch.utils.data.TensorDataset(temp_train_x_e1.float()/255, temp_train_y_e1)
temp_train_dataset_e1.data = temp_train_x_e1.float()/255
temp_train_dataset_e1.targets = temp_train_y_e1
temp_train_dataset_e2 = torch.utils.data.TensorDataset(temp_train_x_e2.float()/255, temp_train_y_e2)
temp_train_dataset_e2.data = temp_train_x_e2.float()/255
temp_train_dataset_e2.targets = temp_train_y_e2

temp_test_dataset_e1 = torch.utils.data.TensorDataset(temp_test_x_e1.float()/255, temp_test_y_e1)
temp_test_dataset_e1.data = temp_test_x_e1.float()/255
temp_test_dataset_e1.targets = temp_test_y_e1
temp_test_dataset_e2 = torch.utils.data.TensorDataset(temp_test_x_e2.float()/255, temp_test_y_e2)
temp_test_dataset_e2.data = temp_test_x_e2.float()/255
temp_test_dataset_e2.targets = temp_test_y_e2

train_loader_e1= DataLoader(temp_train_dataset_e1, batch_size=batch_size_train, shuffle=True)
train_loader_e2= DataLoader(temp_train_dataset_e2, batch_size=batch_size_train, shuffle=True)

test_loader_e1= DataLoader(temp_test_dataset_e1, batch_size=batch_size_test, shuffle=True)
test_loader_e2= DataLoader(temp_test_dataset_e2, batch_size=batch_size_test, shuffle=True)

#####################################################################################################################
"NN Init"

HIDDEN_SIZE=256
X_DIM=28*28
Z_DIM=10
if torch.cuda.is_available():
    torch.cuda.current_device()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
vae = Vae_Irm(x_dim=X_DIM, z_dim=Z_DIM, hidden_size=HIDDEN_SIZE, device=device).to(device)
vae.load_state_dict(torch.load('vae_irm.pth'))
vae.requires_grad_(False)
vae.eval()

#################################################################################################################
"Get Embeddings Training!"

z_1, mu_1, logvar_1 = vae.encode(temp_train_dataset_e1.data.to(device),1)
z_2, mu_2, logvar_2 = vae.encode(temp_train_dataset_e2.data.to(device),2)

z_1 = mu_1
z_2 = mu_2

z_1_I = z_1[:,0:10].cpu()
z_1_e = z_1[:,10:20].cpu()
z_2_I = z_2[:,0:10].cpu()
z_2_e = z_2[:,10:20].cpu()

z_I_train = torch.cat((z_1_I, z_2_I), 0)
z_e_train = torch.cat((z_1_e, z_2_e), 0)

labels_train = torch.cat((temp_train_dataset_e1.targets, temp_train_dataset_e2.targets), 0)

e1_gt = temp_train_dataset_e1.targets*0
e2_gt = temp_train_dataset_e2.targets
e2_gt[:] = 1
e2_gt = e2_gt.long()
env_train = torch.cat((e1_gt, e2_gt), 0)
#################################################################################################################
"Get Embeddings Test!"

z_1, mu_1, logvar_1 = vae.encode(temp_test_dataset_e1.data.to(device),1)
z_2, mu_2, logvar_2 = vae.encode(temp_test_dataset_e2.data.to(device),2)

z_1 = mu_1
z_2 = mu_2

z_1_I = z_1[:,0:10].cpu()
z_1_e = z_1[:,10:20].cpu()
z_2_I = z_2[:,0:10].cpu()
z_2_e = z_2[:,10:20].cpu()

z_I_test = torch.cat((z_1_I, z_2_I), 0)
z_e_test = torch.cat((z_1_e, z_2_e), 0)

labels_test = torch.cat((temp_test_dataset_e1.targets, temp_test_dataset_e2.targets), 0)

e1_gt = temp_test_dataset_e1.targets*0
e2_gt = temp_test_dataset_e2.targets
e2_gt[:] = 1
e2_gt = e2_gt.long()
env_test = torch.cat((e1_gt, e2_gt), 0)
##############################################################################################################
"Baseline Training"

X_DIM = 28*28*3

X_train =  torch.cat((temp_train_dataset_e1.data, temp_train_dataset_e2.data), 0).squeeze().view(-1, X_DIM)
X_test = torch.cat((temp_test_dataset_e1.data, temp_test_dataset_e2.data), 0).squeeze().view(-1, X_DIM)

W_B2L = linear_model.LogisticRegression()
W_B2E = linear_model.LogisticRegression()

W_B2L.fit(X_train, labels_train)
W_B2E.fit(X_train, env_train)

##############################################################################################################
"Training"

W_I2L = linear_model.LogisticRegression()
W_I2E = linear_model.LogisticRegression()
W_E2L = linear_model.LogisticRegression()
W_E2E = linear_model.LogisticRegression()

W_I2L.fit(z_I_train, labels_train)
W_I2E.fit(z_I_train, env_train)
W_E2L.fit(z_e_train, labels_train)
W_E2E.fit(z_e_train, env_train)

###################################################################################################################################################
'Test Loss and Classification Accuracy'

W_B2L_pred = W_B2L.predict(X_test)
W_B2E_pred = W_B2E.predict(X_test)

W_I2L_pred = W_I2L.predict(z_I_test)
W_I2E_pred = W_I2E.predict(z_I_test)
W_E2L_pred = W_E2L.predict(z_e_test)
W_E2E_pred = W_E2E.predict(z_e_test)

print("B2L Test Accuracy:", accuracy_score(labels_test, W_B2L_pred))
print("B2E Test Accuracy:", accuracy_score(env_test, W_B2E_pred))

print("I2L Test Accuracy:", accuracy_score(labels_test, W_I2L_pred))
print("I2E Test Accuracy:", accuracy_score(env_test, W_I2E_pred))
print("E2L Test Accuracy:", accuracy_score(labels_test, W_E2L_pred))
print("E2E Test Accuracy:", accuracy_score(env_test, W_E2E_pred))
