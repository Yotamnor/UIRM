import numpy as np
import time
from collections import OrderedDict

# pytorch imports
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from domainbed import networks, hparams_registry

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


# encoder - Q(z|X)
class VaeEncoderEnv1(torch.nn.Module):
    """
       This class builds the encoder for the VAE
       :param x_dim: input dimensions
       :param hidden_size: hidden layer size
       :param z_dim: latent dimensions
       :param device: cpu or gpu
       """

    def __init__(self, hparams = hparams_registry.default_hparams(algorithm = 'ERM', dataset= 'PACS'), input_shape = (3,64,64), z_dim=8, device=torch.device("cpu")):
        super(VaeEncoderEnv1, self).__init__()

        self.z_dim = z_dim
        self.device = device
        self.input_shape = input_shape


        ###############################################3
        self.latent_dim = z_dim
        latent_dim = z_dim
        modules = []
        in_channels = 3
        # hidden_dims = [32, 64, 128, 256, 512]
        hidden_dims = [32, 32, 32, 32, 32]

        # Build Encoder
        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels=h_dim,
                              kernel_size=3, stride=2, padding=1),
                    nn.BatchNorm2d(h_dim),
                    nn.LeakyReLU())
            )
            in_channels = h_dim

        self.encoder = nn.Sequential(*modules)
        self.fc_mu = nn.Linear(hidden_dims[-1] * 4, latent_dim)
        self.fc_var = nn.Linear(hidden_dims[-1] * 4, latent_dim)

    def bottleneck(self, h):
        """
        This function takes features from the encoder and outputs mu, log-var and a latent space vector z
        :param h: features from the encoder
        :return: z, mu, log-variance
        """
        # mu, logvar = self.fc1(h), self.fc2(h)
        # use the reparametrization trick as torch.normal(mu, logvar.exp()) is not differentiable

        ##########################################################################################
        """
        Encodes the input by passing through the encoder network
        and returns the latent codes.
        :param input: (Tensor) Input tensor to encoder [N x C x H x W]
        :return: (Tensor) List of latent codes
        """
        result = self.encoder(h)
        result = torch.flatten(result, start_dim=1)

        # Split the result into mu and var components
        # of the latent Gaussian distribution
        mu = self.fc_mu(result)
        logvar = self.fc_var(result)

        ########################################################################################

        z = reparameterize(mu, logvar, device=self.device)
        return z, mu, logvar

    def forward(self, x):
        """
        This is the function called when doing the forward pass:
        z, mu, logvar = VaeEncoder(X)
        """
        # h = self.features(x)
        # h = F.relu(h)
        z, mu, logvar = self.bottleneck(x)
        return z, mu, logvar


# encoder - Q(z|X)
class VaeEncoderEnv2(torch.nn.Module):
    """
       This class builds the encoder for the VAE
       :param x_dim: input dimensions
       :param hidden_size: hidden layer size
       :param z_dim: latent dimensions
       :param device: cpu or gpu
       """

    def __init__(self, hparams=hparams_registry.default_hparams(algorithm='ERM', dataset='PACS'),
                 input_shape=(3, 224, 224), z_dim=8, device=torch.device("cpu")):
        super(VaeEncoderEnv2, self).__init__()

        self.z_dim = z_dim
        self.device = device
        self.input_shape = input_shape

        ###############################################3
        self.latent_dim = z_dim
        latent_dim = z_dim
        modules = []
        in_channels = 3
        # hidden_dims = [32, 64, 128, 256, 512]
        hidden_dims = [32, 32, 32, 32, 32]

        # Build Encoder
        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels=h_dim,
                              kernel_size=3, stride=2, padding=1),
                    nn.BatchNorm2d(h_dim),
                    nn.LeakyReLU())
            )
            in_channels = h_dim

        self.encoder = nn.Sequential(*modules)
        self.fc_mu = nn.Linear(hidden_dims[-1] * 4, latent_dim)
        self.fc_var = nn.Linear(hidden_dims[-1] * 4, latent_dim)

    def bottleneck(self, h):
        """
        This function takes features from the encoder and outputs mu, log-var and a latent space vector z
        :param h: features from the encoder
        :return: z, mu, log-variance
        """
        # mu, logvar = self.fc1(h), self.fc2(h)
        # use the reparametrization trick as torch.normal(mu, logvar.exp()) is not differentiable

        ##########################################################################################
        """
        Encodes the input by passing through the encoder network
        and returns the latent codes.
        :param input: (Tensor) Input tensor to encoder [N x C x H x W]
        :return: (Tensor) List of latent codes
        """
        result = self.encoder(h)
        result = torch.flatten(result, start_dim=1)

        # Split the result into mu and var components
        # of the latent Gaussian distribution
        mu = self.fc_mu(result)
        logvar = self.fc_var(result)

        ########################################################################################

        z = reparameterize(mu, logvar, device=self.device)
        return z, mu, logvar

    def forward(self, x):
        """
        This is the function called when doing the forward pass:
        z, mu, logvar = VaeEncoder(X)
        """
        # h = self.features(x)
        # h = F.relu(h)
        z, mu, logvar = self.bottleneck(x)
        return z, mu, logvar

class VaeEncoderCausal(torch.nn.Module):
    """
       This class builds the encoder for the VAE
       :param x_dim: input dimensions
       :param hidden_size: hidden layer size
       :param z_dim: latent dimensions
       :param device: cpu or gpu
       """

    def __init__(self, hparams = hparams_registry.default_hparams(algorithm = 'ERM', dataset= 'PACS'), input_shape = (3,224,224), z_c_dim=56, z_e_dim = 8, device=torch.device("cpu")):
        super(VaeEncoderCausal, self).__init__()
        self.z_c_dim = z_c_dim
        self.z_e_dim = z_e_dim
        self.device = device
        self.input_shape = input_shape

        # self.features = networks.Featurizer(input_shape, hparams)

        ###############################################3
        self.latent_dim = z_c_dim
        latent_dim = z_c_dim
        modules = []
        in_channels = 3
        hidden_dims = [32, 64, 128, 256, 512]


        # Build Encoder
        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels=h_dim,
                              kernel_size=3, stride=2, padding=1),
                    nn.BatchNorm2d(h_dim),
                    nn.LeakyReLU())
            )
            in_channels = h_dim

        self.encoder = nn.Sequential(*modules)
        self.fc_mu_1 = nn.Linear(hidden_dims[-1] * 4 + z_e_dim, hidden_dims[-1] * 4)
        self.fc_mu_2 = nn.Linear(hidden_dims[-1] * 4, latent_dim)

        self.fc_var_1 = nn.Linear(hidden_dims[-1] * 4 + z_e_dim, hidden_dims[-1] * 4)
        self.fc_var_2 = nn.Linear(hidden_dims[-1] * 4, latent_dim)
        ###############################################3

        # self.fc1 = nn.Linear(self.features.n_outputs + z_dim, self.z_dim, bias=True)  # fully-connected to output mu
        # self.fc2 = nn.Linear(self.features.n_outputs + z_dim, self.z_dim, bias=True)  # fully-connected to output logvar

    # def features1(self, x, z_e):
    #     h = F.relu(self.features(x))
    #     h = torch.cat((h, z_e), dim=1)
    #     return h

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
        ## h = self.features1(torch.cat((self.features0(x), z_e), dim=1))
        # h = self.features1(x, z_e)#.detach())
        # z, mu, logvar = self.bottleneck(h)
        ###########################################
        """
        Encodes the input by passing through the encoder network
        and returns the latent codes.
        :param input: (Tensor) Input tensor to encoder [N x C x H x W]
        :return: (Tensor) List of latent codes
        """
        result = self.encoder(x)
        result = torch.flatten(result, start_dim=1)

        # Split the result into mu and var components
        # of the latent Gaussian distribution
        # z_e_detach = z_e.detach()
        mu_features = self.fc_mu_1(torch.cat((result, z_e), dim=1))
        mu = self.fc_mu_2(F.leaky_relu(mu_features))
        logvar_features = self.fc_var_1(torch.cat((result, z_e), dim=1))
        logvar = self.fc_var_2(F.leaky_relu(logvar_features))
        z = reparameterize(mu, logvar, device=self.device)
        ################################################
        return z, mu, logvar

class VaeDecoder(torch.nn.Module):
    """
       This class builds the decoder for the VAE
       :param x_dim: input dimensions
       :param hidden_size: hidden layer size
       :param z_dim: latent dimensions
       """

    def __init__(self, hparams = hparams_registry.default_hparams(algorithm = 'ERM', dataset= 'PACS'), z_dim=64, device=torch.device("cpu")):
        super(VaeDecoder, self).__init__()

        self.device = device
        self.z_dim = z_dim

        hidden_dims = [32, 64, 128, 256, 512]
        latent_dim= z_dim
        modules = []

        self.decoder_input = nn.Linear(latent_dim, hidden_dims[-1] * 4)

        hidden_dims.reverse()

        for i in range(len(hidden_dims) - 1):
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(hidden_dims[i],
                                       hidden_dims[i + 1],
                                       kernel_size=3,
                                       stride=2,
                                       padding=1,
                                       output_padding=1),
                    nn.BatchNorm2d(hidden_dims[i + 1]),
                    nn.LeakyReLU())
            )

        self.decoder = nn.Sequential(*modules)

        self.final_layer = nn.Sequential(
            nn.ConvTranspose2d(hidden_dims[-1],
                               hidden_dims[-1],
                               kernel_size=3,
                               stride=2,
                               padding=1,
                               output_padding=1),
            nn.BatchNorm2d(hidden_dims[-1]),
            nn.LeakyReLU(),
            nn.Conv2d(hidden_dims[-1], out_channels=3,
                      kernel_size=3, padding=1),
            nn.Sigmoid())

    def forward(self, z):
        """
        This is the function called when doing the forward pass:
        x_reconstruction = VaeDecoder(z)
        """
        # z_sq = z.view(-1,2,10,10)
        # x = self.decoder(z_sq)
        # x = x.view(-1, *self.shape)
        h = self.decoder_input(z)
        h = h.view(-1, 512, 2, 2)
        h = self.decoder(h)
        x = self.final_layer(h)
        return x

class Vae_Irm(torch.nn.Module):
    def __init__(self, input_shape = (3,64,64), z_c_dim=56, z_e_dim=8, device=torch.device("cpu")):
        super(Vae_Irm, self).__init__()
        self.device = device
        self.z_c_dim = z_c_dim
        self.z_e_dim = z_e_dim
        hidden_size = input_shape[0]*input_shape[1]*input_shape[2]
        self.encoder_env1 = VaeEncoderEnv1(device= self.device)
        self.encoder_env2 = VaeEncoderEnv2(device= self.device)
        self.encoder_causal = VaeEncoderCausal(device= self.device)
        self.decoder = VaeDecoder(device= self.device)

    def encode(self, x, e):
        if (e==1):
            z_e, mu_e, logvar_e = self.encoder_env1(x)
        if (e==2):
            z_e, mu_e, logvar_e = self.encoder_env2(x)

        z_c, mu_c, logvar_c = self.encoder_causal(x, z_e)
        z = torch.cat((z_c, z_e), dim=1)
        mu = torch.cat((mu_c, mu_e), dim=1)
        logvar = torch.cat((logvar_c, logvar_e), dim=1)
        # z = torch.cat((0, z_e), dim=1)
        # mu = torch.cat((0, mu_e), dim=1)
        # logvar = torch.cat((0, logvar_e), dim=1)
        return z, mu, logvar

    def decode(self, z):
        x = self.decoder(z)
        return x

    def sample(self, num_samples=1, bias=torch.zeros(64), freeze = 0, z_pre = []):
        """
        This functions generates new data by sampling random variables and decoding them.
        Vae.sample() actually generatess new data!
        Sample z ~ N(0,1)
        """
        if freeze == 0:
            z = torch.randn(num_samples, self.z_c_dim + self.z_e_dim).to(self.device)
            z = z + bias.to(self.device)
            # z = z*bias.to(self.device)
        elif freeze == 1:
            z_c = torch.randn(1, self.z_c_dim).to(self.device)
            z_e = torch.randn(num_samples, self.z_e_dim).to(self.device)
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


def beta_loss_function(recon_x, x, mu, logvar, loss_type='bce', bias = torch.zeros(64), beta=1):
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



def train_beta_vae(beta,dataloader_e1,dataloader_e2,BATCH_SIZE=128,LEARNING_RATE=1e-3,NUM_EPOCHS=50,HIDDEN_SIZE=256,X_DIM=28*28, vae=Vae_Irm(), fine_tune=0):
    # training

    # check if there is gpu avilable, if there is, use it
    if torch.cuda.is_available():
        torch.cuda.current_device()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu")
    print("running calculations on: ", device)


    if fine_tune==0:
        vae = Vae_Irm(device=device).to(device)

    # optimizer
    vae_optim = torch.optim.Adam(params=vae.parameters(), lr=LEARNING_RATE)
    # scheduler = torch.optim.lr_scheduler.StepLR(vae_optim, step_size=25, gamma=0.5)

    ####################################################
    from torch.optim.lr_scheduler import LambdaLR, StepLR

    # Define your custom LR schedule
    def lr_lambda(epoch):
        if epoch < 10:
            return epoch / 10.0  # Linear warm-up (0 → 1 over 10 epochs)
        elif epoch < 25:
            return 1.0
        elif epoch < 50:
            return 0.25
        elif epoch < 75:
            return 0.25*0.25
        elif epoch < 100:
            return 0.25*0.25*0.1
        elif epoch < 125:
            return 0.25*0.25*0.25*0.25
        elif epoch < 150:
            return 0.25*0.25*0.25*0.25*0.25
        elif epoch < 175:
            return 0.25*0.25*0.25*0.25*0.25*0.25
        else:
            return 0.25 * 0.25 * 0.25 * 0.25 * 0.25 * 0.25 * 0.1


    scheduler = LambdaLR(vae_optim, lr_lambda=lr_lambda)
    #################################################

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
                bias = torch.zeros(64).to(device)
                bias[62]=100
            else:
                x= x_e2
                e= 2
                bias = torch.zeros(64).to(device)
                bias[63]=100
            # if (batch_i <= env_indx):
            x_recon, mu, logvar, z = vae(x,e)
            # calculate the loss
            beta_ac = min(1, beta*(epoch))
            recon_error,kl,loss = beta_loss_function(x_recon, x, mu, logvar, loss_type='bce', bias= bias, beta=beta_ac)
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

        # if np.mod(epoch+1 , 25) == 0:
        #     vae_optim.param_groups[0]['lr'] = vae_optim.param_groups[0]['lr']/2

        # if epoch > 100:
        #     vae_optim.param_groups[0]['lr'] = vae_optim.param_groups[0]['lr']*0.95
        print("epoch: {} recon_error: {:.5f} kl: {:.5f} training loss: {:.5f} epoch time: {:.3f} sec".format(epoch,train_recon_errors[-1],train_kls[-1], train_losses[-1],
                                                                                                             time.time() - epoch_start_time))
        # torch.cuda.empty_cache()

        scheduler.step()

    # save
    fname = "./beta_{:.2f}_vae_{}_epochs.pth".format(beta,NUM_EPOCHS)
    torch.save(vae.state_dict(), fname)
    print("saved checkpoint @", fname)
    return vae,train_recon_errors,train_kls,train_losses
