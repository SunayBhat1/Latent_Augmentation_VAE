import torch
import torch.nn as nn                         
import torch.nn.functional as F
from torch.utils import data

import numpy as np


#####  Latent Augmentation VAE   ####################################

### Subclasses ###

class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, layers=[16,16], activation=F.relu):
        '''
        Multi-layer perceptron
        input_dim: dimension of input
        output_dim: dimension of output
        layers: list of hidden layer dimensions
        activation: activation function
        '''
        super(MLP, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.layers = layers
        self.activation = activation

        self.fc_in = nn.Linear(input_dim, self.layers[0])
        hidden_layers = []
        for i in range(len(self.layers) - 1):
            hidden_layers.append(nn.Linear(in_features=self.layers[i], out_features=self.layers[i + 1]))
        self.hidden_layers = nn.Sequential(*hidden_layers)
        self.fc_out = nn.Linear(self.layers[-1], output_dim)

    def forward(self, x):
        x = self.activation(self.fc_in(x))
        x = self.hidden_layers(x)
        x = self.fc_out(x)
        return x

class Encoder(nn.Module):
    def __init__(self, latent_dim, num_filters, image_dim=28):
        '''
        Image Encoder Network
        num_filters: number of filters in convolutional layers
        latent_dim: dimension of latent space
        image_dim: dimension of input image
        '''
        super(Encoder, self).__init__()

        self.num_filters = num_filters
        self.latent_dims = latent_dim
        self.image_dim = image_dim

        filter_dim_1 = self.image_dim // 4

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=num_filters, kernel_size=4, stride=2, padding=1) # out: c x image_dim/2 x image_dim/2
        self.conv2 = nn.Conv2d(in_channels=num_filters, out_channels=num_filters*2, kernel_size=4, stride=2, padding=1) # out: c x image_dim/4 x image_dim/4
        self.fc_mu = nn.Linear(in_features=num_filters*2*filter_dim_1*filter_dim_1, out_features=latent_dim)
        self.fc_logvar = nn.Linear(in_features=num_filters*2*filter_dim_1*filter_dim_1, out_features=latent_dim)
            
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(x.size(0), -1) # flatten batch of multi-channel feature maps to a batch of feature vectors
        x_mu = self.fc_mu(x)
        x_logvar = self.fc_logvar(x)
        return x_mu, x_logvar

class Decoder(nn.Module):
    def __init__(self, latent_dim, num_filters, image_dim=28, augs=None):
        '''
        Image Decoder Network
        num_filters: number of filters in convolutional layers
        latent_dim: dimension of latent space
        image_dim: dimension of input image
        '''
        super(Decoder, self).__init__()

        self.num_filters = num_filters
        self.latent_dims = latent_dim
        self.image_dim = image_dim
        self.augs = augs

        self.filter_dim_1 = self.image_dim // 4 # Divide and round down (floor)

        self.fc = nn.Linear(in_features=latent_dim, out_features=num_filters*2*self.filter_dim_1*self.filter_dim_1)
        self.conv2 = nn.ConvTranspose2d(in_channels=num_filters*2, out_channels=num_filters, kernel_size=4, stride=2, padding=1)
        self.conv1 = nn.ConvTranspose2d(in_channels=num_filters, out_channels=1, kernel_size=4, stride=2, padding=1)
            
    def forward(self, x):
        x = self.fc(x)
        x = x.view(x.size(0), self.num_filters*2, self.filter_dim_1, self.filter_dim_1) # unflatten batch of feature vectors to a batch of multi-channel feature maps
        x = F.relu(self.conv2(x))
        x = torch.sigmoid(self.conv1(x)) # last layer before output is sigmoid, since we are using BCE as reconstruction loss
        return x



class LAVAE(nn.Module):
    def __init__(self,
                latent_dim=2,
                num_dec = 1,
                num_filters=256,
                image_dim=28,
                augs=['augs'],
                llinear=True,
                ):
        '''
        Latent Augmented Variational Autoencoder
        num_filters: number of filters in convolutional layers
        latent_dim: dimension of latent space
        image_dim: dimension of input image
        augs: list of latent augmentations
        '''
        super(LAVAE, self).__init__()

        self.latent_dim = latent_dim
        self.trained_augs = augs
        self.num_dec = num_dec
        self.latent_linear = llinear
        self.losses = {}
        
        self.encoder = Encoder(num_filters=num_filters, latent_dim=latent_dim, image_dim=image_dim)
        self.decoders = nn.ModuleList([Decoder(num_filters=num_filters, latent_dim=latent_dim, image_dim=image_dim) for iDec in range(num_dec)])
        self.decoders[0].augs = augs
        if llinear:
            self.Laugs = nn.ParameterList([nn.Parameter(torch.zeros((latent_dim,latent_dim))) for _ in augs])
            for laug in self.Laugs: 
                torch.nn.init.xavier_uniform_(laug) #or any other init method
        else:
            self.Laugs = nn.ModuleDict({aug: MLP(input_dim=latent_dim, output_dim=latent_dim, layers=[8,8]) for aug in augs})
    
    def forward(self, x,iDec=0):
        latent_mu, latent_logvar = self.encoder(x)
        latent = self.latent_sample(latent_mu, latent_logvar)
        x_recon = self.decoders[iDec](latent)
        return x_recon, latent_mu, latent_logvar
    
    def latent_sample(self, mu, logvar,bypass=False):
        if self.training or bypass:
            # the reparameterization trick
            std = logvar.mul(0.5).exp_()
            eps = torch.empty_like(std).normal_()
            return eps.mul(std).add_(mu)
        else:
            return mu

# Used for custom data
class torch_dataloader(data.Dataset):

    def __init__(self, data_array, transform=None,targets=None):
        self.data = data_array
        self.targets = targets
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if self.transform:
            if self.targets is not None:
                return self.transform(self.data[idx]).to(torch.float32), self.targets[idx]
            else:
                return self.transform(self.data[idx]).to(torch.float32)
        else:
            if self.targets is not None:
                return self.data[idx].to(torch.float32), self.targets[idx]
            else:
                return self.data[idx].to(torch.float32)

# Load original and augmented images for Laug trainig
class torch_latent_dataloader(data.Dataset):

    def __init__(self, data_arrays, transform=None):
        self.data_arrays = data_arrays
        self.transform = transform

    def __len__(self):
        return len(self.data_arrays['orig'])

    def __getitem__(self, idx):

        if self.transform:
            return {a: self.transform(data[idx]).to(torch.float32) for a,data in self.data_arrays.items()} #tuple([self.transform(data[idx]) for data in self.data_arrays])
        else:
            return {a: data[idx].to(torch.float32) for a,data in self.data_arrays.items()} #tuple([data[idx] for data in self.data_arrays]) 

