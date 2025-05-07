import numpy as np
import time
from collections import OrderedDict

# pytorch imports
import torch
import torch.nn as nn
import torch.nn.functional as F

# reparametrization trick
def reparameterize(mu, logvar, device=torch.device("cpu")):
    """
    This function applies the reparameterization trick:
    z = mu(X) + sigma(X)^0.5 * epsilon, where epsilon ~ N(0,I)
    :param mu: mean of x
    :param logvar: log variance of x
    :param device: device to perform calculations on
    :return z: the sampled latent variable
    """
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std).to(device)
    return mu + eps * std


class VaeEncoderEnv1(torch.nn.Module):
    """
       This class builds the encoder for the VAE
       :param x_dim: input dimensions
       :param hidden_size: hidden layer size
       :param z_dim: latent dimensions
       :param device: cpu or gpu
       """

    def __init__(self, x_dim=28 * 28, hidden_size=256, z_dim=10, num_in_channels = 3, device=torch.device("cpu")):
        super(VaeEncoderEnv1, self).__init__()
        self.x_dim = x_dim
        self.hidden_size = hidden_size
        self.z_dim = z_dim
        self.device = device

        # self.features = nn.Sequential(nn.Linear(x_dim, self.hidden_size),
        #                               nn.ReLU())
        # self.fc1 = nn.Linear(self.hidden_size, self.z_dim, bias=True)  # fully-connected to output mu
        # self.fc2 = nn.Linear(self.hidden_size, self.z_dim, bias=True)  # fully-connected to output logvar


        self.pre_features = nn.Sequential(OrderedDict([
            ('conv1', nn.Conv2d(num_in_channels, 10, 5)),
            ('bn1', nn.BatchNorm2d(10, momentum=1, affine=True)),
            ('relu1', nn.ReLU(inplace=True)),
            ('pool1', nn.MaxPool2d(2,2)),
            ('conv2', nn.Conv2d(10,20,5)),
            ('bn2', nn.BatchNorm2d(20, momentum=1, affine=True)),
            ('relu2', nn.ReLU(inplace=True)),
            ('pool2', nn.MaxPool2d(2,2)),
            ('conv3', nn.Conv2d(20,100,4)),
            ('relu3', nn.ReLU(inplace=True))
        ]))
        self.fc0 = nn.Linear(100, 50, bias = True)

        self.fc1 = nn.Linear(50, self.z_dim, bias=True)  # fully-connected to output mu
        self.fc2 = nn.Linear(50, self.z_dim, bias=True)  # fully-connected to output logvar

    def features(self, x):
        h = self.pre_features(x)
        h = h.view(-1, 100)
        h = F.relu(self.fc0(h))
        return h

    def bottleneck(self, h):
        """
        This function takes features from the encoder and outputs mu, log-var and a latent space vector z
        :param h: features from the encoder
        :return: z, mu, log-variance
        """
        mu, logvar = self.fc1(h), self.fc2(h)
        # use the reparametrization trick as torch.normal(mu, logvar.exp()) is not differentiable
        z = reparameterize(mu, logvar, device=self.device)
        return z, mu, logvar

    def forward(self, x):
        """
        This is the function called when doing the forward pass:
        z, mu, logvar = VaeEncoder(X)
        """
        h = self.features(x)
        z, mu, logvar = self.bottleneck(h)
        return z, mu, logvar

class VaeEncoderEnv2(torch.nn.Module):
    """
       This class builds the encoder for the VAE
       :param x_dim: input dimensions
       :param hidden_size: hidden layer size
       :param z_dim: latent dimensions
       :param device: cpu or gpu
       """

    def __init__(self, x_dim=28 * 28, hidden_size=256, z_dim=10, num_in_channels = 3, device=torch.device("cpu")):
        super(VaeEncoderEnv2, self).__init__()
        self.x_dim = x_dim
        self.hidden_size = hidden_size
        self.z_dim = z_dim
        self.device = device

        self.pre_features = nn.Sequential(OrderedDict([
            ('conv1', nn.Conv2d(num_in_channels, 10, 5)),
            ('bn1', nn.BatchNorm2d(10, momentum=1, affine=True)),
            ('relu1', nn.ReLU(inplace=True)),
            ('pool1', nn.MaxPool2d(2,2)),
            ('conv2', nn.Conv2d(10,20,5)),
            ('bn2', nn.BatchNorm2d(20, momentum=1, affine=True)),
            ('relu2', nn.ReLU(inplace=True)),
            ('pool2', nn.MaxPool2d(2,2)),
            ('conv3', nn.Conv2d(20,100,4)),
            ('relu3', nn.ReLU(inplace=True))
        ]))
        self.fc0 = nn.Linear(100, 50, bias = True)
        self.fc1 = nn.Linear(50, self.z_dim, bias=True)  # fully-connected to output mu
        self.fc2 = nn.Linear(50, self.z_dim, bias=True)  # fully-connected to output logvar

    def features(self, x):
        h = self.pre_features(x)
        h = h.view(-1, 100)
        h = F.relu(self.fc0(h))
        return h

    def bottleneck(self, h):
        """
        This function takes features from the encoder and outputs mu, log-var and a latent space vector z
        :param h: features from the encoder
        :return: z, mu, log-variance
        """
        mu, logvar = self.fc1(h), self.fc2(h)
        # use the reparametrization trick as torch.normal(mu, logvar.exp()) is not differentiable
        z = reparameterize(mu, logvar, device=self.device)
        return z, mu, logvar

    def forward(self, x):
        """
        This is the function called when doing the forward pass:
        z, mu, logvar = VaeEncoder(X)
        """
        h = self.features(x)
        z, mu, logvar = self.bottleneck(h)
        return z, mu, logvar

class VaeEncoderCausal(torch.nn.Module):
    """
       This class builds the encoder for the VAE
       :param x_dim: input dimensions
       :param hidden_size: hidden layer size
       :param z_dim: latent dimensions
       :param device: cpu or gpu
       """

    def __init__(self, x_dim=28 * 28, z_e_dim = 10, hidden_size=256, z_c_dim=10, num_in_channels = 3, device=torch.device("cpu")):
        super(VaeEncoderCausal, self).__init__()
        self.x_dim = x_dim
        self.hidden_size = hidden_size
        self.z_c_dim = z_c_dim
        self.z_e_dim = z_e_dim
        self.device = device

        self.features0 = nn.Sequential(OrderedDict([
            ('conv1', nn.Conv2d(num_in_channels, 10, 5)),
            ('bn1', nn.BatchNorm2d(10, momentum=1, affine=True)),
            ('relu1', nn.ReLU(inplace=True)),
            ('pool1', nn.MaxPool2d(2,2)),
            ('conv2', nn.Conv2d(10,20,5)),
            ('bn2', nn.BatchNorm2d(20, momentum=1, affine=True)),
            ('relu2', nn.ReLU(inplace=True)),
            ('pool2', nn.MaxPool2d(2,2)),
            ('conv3', nn.Conv2d(20,100,4)),
            ('relu3', nn.ReLU(inplace=True))
        ]))
        self.fc0 = nn.Linear(100 +
                             z_e_dim, 50, bias = True)

        self.fc1 = nn.Linear(50, self.z_c_dim, bias=True)  # fully-connected to output mu
        self.fc2 = nn.Linear(50, self.z_c_dim, bias=True)  # fully-connected to output logvar

    def features1(self, x, z_e):
        h = self.features0(x)
        h = h.view(-1, 100)
        h = torch.cat((h, z_e), dim=1)
        h = F.relu(self.fc0(h))
        return h

    def bottleneck(self, h):
        """
        This function takes features from the encoder and outputs mu, log-var and a latent space vector z
        :param h: features from the encoder
        :return: z, mu, log-variance
        """
        mu, logvar = self.fc1(h), self.fc2(h)
        # use the reparametrization trick as torch.normal(mu, logvar.exp()) is not differentiable
        z = reparameterize(mu, logvar, device=self.device)
        return z, mu, logvar

    def forward(self, x, z_e):
        """
        This is the function called when doing the forward pass:
        z, mu, logvar = VaeEncoder(X)
        """
        # h = self.features1(torch.cat((self.features0(x), z_e), dim=1))
        h = self.features1(x, z_e)
        z, mu, logvar = self.bottleneck(h)
        return z, mu, logvar

class VaeDecoder(torch.nn.Module):
    """
       This class builds the decoder for the VAE
       :param x_dim: input dimensions
       :param hidden_size: hidden layer size
       :param z_dim: latent dimensions
       """

    def __init__(self, x_dim=28 * 28, hidden_size=256, z_dim=20, num_out_channels = 3):
        super(VaeDecoder, self).__init__()
        self.x_dim = x_dim
        self.hidden_size = hidden_size
        self.z_dim = z_dim
        self.num_out_channels = num_out_channels

        self.decoder = nn.Sequential(nn.Linear(self.z_dim, self.hidden_size),
                                     # nn.BatchNorm1d(self.hidden_size, momentum=1, affine=True),
                                     # nn.ReLU(),
                                     nn.LeakyReLU(),
                                     nn.Linear(self.hidden_size, self.hidden_size*num_out_channels),
                                     # nn.BatchNorm1d(self.hidden_size, momentum=1, affine=True),
                                     # nn.ReLU(),
                                     nn.LeakyReLU(),
                                     nn.Linear(self.hidden_size*num_out_channels, self.x_dim*num_out_channels),
                                     nn.Sigmoid())

    def forward(self, z):
        """
        This is the function called when doing the forward pass:
        x_reconstruction = VaeDecoder(z)
        """
        x = self.decoder(z)
        x_dim_img = 28
        x = x.view(-1, self.num_out_channels, x_dim_img, x_dim_img)
        return x

class Vae_Irm(torch.nn.Module):
    def __init__(self, x_dim=28 * 28, z_dim=10, hidden_size=256, device=torch.device("cpu")):
        super(Vae_Irm, self).__init__()
        self.device = device
        self.z_dim = z_dim
        self.encoder_env1 = VaeEncoderEnv1(x_dim, hidden_size, z_dim=z_dim, device=device)
        self.encoder_env2 = VaeEncoderEnv2(x_dim, hidden_size, z_dim=z_dim, device=device)
        self.encoder_causal = VaeEncoderCausal(x_dim, z_e_dim=z_dim, hidden_size=hidden_size, z_c_dim=z_dim, device=device)
        self.decoder = VaeDecoder(x_dim, hidden_size, z_dim=2*z_dim)

    def encode(self, x, e):
        if (e==1):
            z_e, mu_e, logvar_e = self.encoder_env1(x)
        if (e==2):
            z_e, mu_e, logvar_e = self.encoder_env2(x)

        z_c, mu_c, logvar_c = self.encoder_causal(x, z_e)
        z = torch.cat((z_c, z_e), dim=1)
        mu = torch.cat((mu_c, mu_e), dim=1)
        logvar = torch.cat((logvar_c, logvar_e), dim=1)
        return z, mu, logvar

    def decode(self, z):
        x = self.decoder(z)
        return x

    def sample(self, num_samples=1, bias=torch.zeros(20), freeze = 0, z_pre = []):
        """
        This functions generates new data by sampling random variables and decoding them.
        Vae.sample() actually generatess new data!
        Sample z ~ N(0,1)
        """
        if freeze == 0:
            z = torch.randn(num_samples, self.z_dim*2).to(self.device)
            z = z+ bias.to(self.device)
            # z = z*bias.to(self.device)
        elif freeze == 1:
            z_c = torch.randn(1, self.z_dim).to(self.device)
            z_e = torch.randn(num_samples, self.z_dim).to(self.device)
            z_c_repeated = z_c.repeat(num_samples, 1)
            z = torch.cat((z_c_repeated, z_e), dim=1)
            z = z+ bias.to(self.device)
            # z = z*bias.to(self.device)
        else:
            z = z_pre.to(self.device)

        return self.decode(z)

    def forward(self, x, e):
        """
        This is the function called when doing the forward pass:
        return x_recon, mu, logvar, z = Vae(X)
        """
        z, mu, logvar = self.encode(x, e)
        x_recon = self.decode(z)
        return x_recon, mu, logvar, z


def loss_function(recon_x, x, mu, logvar, loss_type='bce'):
    """
    This function calculates the loss of the VAE.
    loss = reconstruction_loss - 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    :param recon_x: the reconstruction from the decoder
    :param x: the original input
    :param mu: the mean given X, from the encoder
    :param logvar: the log-variance given X, from the encoder
    :param loss_type: type of loss function - 'mse', 'l1', 'bce'
    :return: VAE loss
    """
    if loss_type == 'mse':
        recon_error = F.mse_loss(recon_x, x, reduction='sum')
    elif loss_type == 'l1':
        recon_error = F.l1_loss(recon_x, x, reduction='sum')
    elif loss_type == 'bce':
        recon_error = F.binary_cross_entropy(recon_x, x, reduction='sum')
    else:
        raise NotImplementedError

    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return (recon_error + kl) / x.size(0)

def beta_loss_function(recon_x, x, mu, logvar, loss_type='bce', bias = torch.zeros(20), beta=1):
    """
    This function calculates the loss of the beta-VAE.
    loss = reconstruction_loss - beta*0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    :param recon_x: the reconstruction from the decoder
    :param x: the original input
    :param mu: the mean given X, from the encoder
    :param logvar: the log-variance given X, from the encoder
    :param loss_type: type of loss function - 'mse', 'l1', 'bce'
    :param beta: kl divergence factor.
    :return: VAE loss
    """
    # x = x.view(-1, 2*28*28)
    if loss_type == 'mse':
        recon_error = F.mse_loss(recon_x, x, reduction='sum')
    elif loss_type == 'l1':
        recon_error = F.l1_loss(recon_x, x, reduction='sum')
    elif loss_type == 'bce':
        recon_error = F.binary_cross_entropy(recon_x, x, reduction='sum')
    else:
        raise NotImplementedError
    recon_error=recon_error/ x.size(0)
    kl = -0.5 * torch.sum(1 + logvar - (mu-bias).pow(2) - logvar.exp())
    kl=kl/x.size(0)
    loss=(recon_error + beta*kl)
    return recon_error.data.cpu().numpy(),kl.data.cpu().numpy(),loss



def train_beta_vae(beta,dataloader_e1,dataloader_e2,BATCH_SIZE=128,LEARNING_RATE=1e-3,NUM_EPOCHS=50,HIDDEN_SIZE=256,X_DIM=28*28,Z_DIM=10, vae=Vae_Irm(), fine_tune=0):
    # training

    # check if there is gpu avilable, if there is, use it
    if torch.cuda.is_available():
        torch.cuda.current_device()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu")
    print("running calculations on: ", device)

    # load the data
    # dataloader = DataLoader(train_data, env_ls, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    # dataloader = train_data
    # create our model and send it to the device (cpu/gpu)

    if fine_tune==0:
        vae = Vae_Irm(x_dim=X_DIM, z_dim=Z_DIM, hidden_size=HIDDEN_SIZE, device=device).to(device)

    # optimizer
    vae_optim = torch.optim.Adam(params=vae.parameters(), lr=LEARNING_RATE)

    # save the losses from each epoch, we might want to plot it later
    train_recon_errors=[]
    train_kls=[]
    train_losses = []

    # here we go
    for epoch in range(NUM_EPOCHS):
        epoch_start_time = time.time()
        batch_recon_errors = []
        batch_kls = []
        batch_losses = []

        for batch_i, batch_e1, batch_e2 in zip(np.arange(len(dataloader_e1)), dataloader_e1, dataloader_e2):
            # forward pass
            # x_e1 = batch_e1[0].to(device).view(-1, X_DIM)  # just the images
            # x_e2 = batch_e2[0].to(device).view(-1, X_DIM)  # just the images
            x_e1 = batch_e1[0].to(device)#.unsqueeze(1)#.view(-1, X_DIM)  # just the images
            x_e2 = batch_e2[0].to(device)#.unsqueeze(1)#.view(-1, X_DIM)  # just the images
            env_pick= np.random.rand(1)
            if env_pick<0.5:
                x= x_e1
                e= 1
                bias = torch.zeros(20).to(device)
                bias[18]=100
            else:
                x= x_e2
                e= 2
                bias = torch.zeros(20).to(device)
                bias[19]=100
            # if (batch_i <= env_indx):
            x_recon, mu, logvar, z = vae(x,e)
            # calculate the loss
            recon_error,kl,loss = beta_loss_function(x_recon, x, mu, logvar, loss_type='bce', bias= bias, beta=beta)
            # optimization (same 3 steps everytime)
            vae_optim.zero_grad()
            loss.backward()
            vae_optim.step()
            # save loss
            batch_recon_errors.append(recon_error)
            batch_kls.append(kl)
            batch_losses.append(loss.data.cpu().item())
        train_recon_errors.append(np.mean(batch_recon_errors))
        train_kls.append(np.mean(batch_kls))
        train_losses.append(np.mean(batch_losses))

        if np.mod(epoch+1 , 25) == 0:
            vae_optim.param_groups[0]['lr'] = vae_optim.param_groups[0]['lr']/2

        if epoch > 90:
            vae_optim.param_groups[0]['lr'] = vae_optim.param_groups[0]['lr']*0.8
        print("epoch: {} recon_error: {:.5f} kl: {:.5f} training loss: {:.5f} epoch time: {:.3f} sec".format(epoch,train_recon_errors[-1],train_kls[-1], train_losses[-1],
                                                                                                             time.time() - epoch_start_time))
    # save
    fname = "./beta_{:.2f}_vae_{}_epochs.pth".format(beta,NUM_EPOCHS)
    torch.save(vae.state_dict(), fname)
    print("saved checkpoint @", fname)
    return vae,train_recon_errors,train_kls,train_losses
