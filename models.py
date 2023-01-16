import torch
import torch.nn as nn                         
import torch.nn.functional as F
from torch.utils import data

import numpy as np


class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, layers=[16,16], activation=F.relu):
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
    def __init__(self,num_filters,latent_dim):
        super(Encoder, self).__init__()

        self.num_filters = num_filters
        self.latent_dims = latent_dim

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=num_filters, kernel_size=4, stride=2, padding=1) # out: c x 16 x 16
        self.conv2 = nn.Conv2d(in_channels=num_filters, out_channels=num_filters*2, kernel_size=4, stride=2, padding=1) # out: c x 8 x 8
        self.fc_mu = nn.Linear(in_features=num_filters*2*8*8, out_features=latent_dim)
        self.fc_logvar = nn.Linear(in_features=num_filters*2*8*8, out_features=latent_dim)
            
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(x.size(0), -1) # flatten batch of multi-channel feature maps to a batch of feature vectors
        x_mu = self.fc_mu(x)
        x_logvar = self.fc_logvar(x)
        return x_mu, x_logvar

class Decoder(nn.Module):
    def __init__(self,num_filters,latent_dim):
        super(Decoder, self).__init__()

        self.num_filters = num_filters
        self.latent_dims = latent_dim

        self.fc = nn.Linear(in_features=latent_dim, out_features=num_filters*2*8*8)
        self.conv2 = nn.ConvTranspose2d(in_channels=num_filters*2, out_channels=num_filters, kernel_size=4, stride=2, padding=1)
        self.conv1 = nn.ConvTranspose2d(in_channels=num_filters, out_channels=1, kernel_size=4, stride=2, padding=1)
            
    def forward(self, x):
        x = self.fc(x)
        x = x.view(x.size(0), self.num_filters*2, 8, 8) # unflatten batch of feature vectors to a batch of multi-channel feature maps
        x = F.relu(self.conv2(x))
        x = torch.sigmoid(self.conv1(x)) # last layer before output is sigmoid, since we are using BCE as reconstruction loss
        return x
    
class LTVAE(nn.Module):
    def __init__(self,
                num_filters=64,
                latent_dim=2,):
        super(LTVAE, self).__init__()
        
        self.encoder = Encoder(num_filters=num_filters, latent_dim=latent_dim)
        self.decoder = Decoder(num_filters=num_filters, latent_dim=latent_dim)
        self.Laug = MLP(input_dim=latent_dim, output_dim=latent_dim, layers=[8,8])
    
    def forward(self, x):
        latent_mu, latent_logvar = self.encoder(x)
        latent = self.latent_sample(latent_mu, latent_logvar)
        x_recon = self.decoder(latent)
        return x_recon, latent_mu, latent_logvar
    
    def latent_sample(self, mu, logvar):
        if self.training:
            # the reparameterization trick
            std = logvar.mul(0.5).exp_()
            eps = torch.empty_like(std).normal_()
            return eps.mul(std).add_(mu)
        else:
            return mu

class torch_dataloader(data.Dataset):

    def __init__(self, data_array):
        self.data = data_array

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

class torch_latent_dataloader(data.Dataset):

    def __init__(self, data_array,flipped_array):
        self.data = data_array
        self.flipped_data = flipped_array

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx],self.flipped_data[idx]

