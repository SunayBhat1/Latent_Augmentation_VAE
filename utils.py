import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.datasets as torch_datasets

from models import torch_dataloader, torch_latent_dataloader

##### Data Utils #####

# Make checkerbaord images of a given size
def checkerBW(N,image_dim,low=0.001,high=0.999):

    black = np.random.uniform(0,low, (N*2, int(image_dim/2*image_dim/2)))
    white = np.random.uniform(high,1, (N*2, int(image_dim/2*image_dim/2)))

    x_top = np.concatenate((black[0:N], white[0:N]), axis=1).reshape(N, image_dim,int(image_dim/2))
    x_bottom = np.concatenate((white[N:2*N], black[N:2*N]), axis=1).reshape(N, image_dim,int(image_dim/2))

    x_checker = np.concatenate((x_top, x_bottom), axis=2)

    return x_checker

# Make checkerboard images of various sizes
def recursivce_checker(N,image_dim=32,checker_size=16,low=0.001,high=0.999):

    depth = int(image_dim/(checker_size*2))

    if depth == 1:
        return checkerBW(N,image_dim,low=low,high=high)

    else:
        x_checker = np.zeros((N,image_dim,image_dim))

        window = checker_size * 2
        for i in range(depth):
            for j in range(depth):
                x_checker[:,i*window:(i+1)*window,j*window:(j+1)*window] = checkerBW(N,checker_size*2,low=low,high=high)

        return x_checker

# Checker Board Images Dataset
def make_checker_dataset(N=1000,image_dim=32,checker_sizes=[16,8,4,2],flipped=True,low=0.001,high=0.999):

    x = recursivce_checker(N,image_dim=image_dim,checker_size=checker_sizes[0],low=low,high=high)

    for i in range(1,len(checker_sizes)):
        x = np.concatenate((x, recursivce_checker(N,image_dim=image_dim,checker_size=checker_sizes[i],low=low,high=high)), axis=0)

    if flipped:
        x_2 = np.flip(x, axis=1)
        x = np.concatenate((x, x_2), axis=0)

    y = np.zeros((x.shape[0], len(checker_sizes)))

    # for i in range(len(checker_sizes)):
    #     y[0:N,i] = 0
    #     if flipped:
    #         y[N:2*N,i] = 1

    return torch.from_numpy(x).float()

# Make black and white images for testing
def make_black_white_images(N=1000,image_dim=28,flipped=False,checker=False):

    black = np.random.uniform(0,0.1, (N, int(image_dim*image_dim/2)))
    white = np.random.uniform(0.9,1, (N, int(image_dim*image_dim/2)))

    x = np.concatenate((black, white), axis=1)

    # Roate images 90 degrees
    x = np.rot90(x.reshape(N, image_dim, image_dim), 1, (1,2)).reshape(N, image_dim*image_dim)

    # get quarter checherboard images
    if checker:
        black = np.random.uniform(0,0.1, (N*2, int(image_dim/2*image_dim/2)))
        white = np.random.uniform(0.9,1, (N*2, int(image_dim/2*image_dim/2)))

        x_top = np.concatenate((black[0:N], white[0:N]), axis=1).reshape(N, image_dim,int(image_dim/2))
        x_bottom = np.concatenate((white[N:2*N], black[N:2*N]), axis=1).reshape(N, image_dim,int(image_dim/2))

        x_checker = np.concatenate((x_top, x_bottom), axis=2)

        if flipped:
            x_checker = np.concatenate((x_checker, np.flip(x_checker, axis=1)), axis=0)
        

    # Flip images
    if flipped:
        x_2 = np.flip(x, axis=1)
        x = np.concatenate((x, x_2), axis=0)
        x = x.reshape(N*2, image_dim,image_dim)
    else:
        x = x.reshape(N, image_dim,image_dim)

    if checker:
        x = np.concatenate((x, x_checker), axis=0)

    # Image labels
    if flipped and checker:
        y = np.zeros((x.shape[0], 4))
        y[0:N,0] = 1
        y[N:2*N,1] = 1
        y[2*N:3*N,2] = 1
        y[3*N:4*N,3] = 1
    elif flipped:
        y = np.zeros((x.shape[0], 2))
        y[0:N,0] = 1
        y[N:2*N,1] = 1
    elif checker:
        y = np.zeros((x.shape[0], 2))
        y[0:N,0] = 1
        y[N:2*N,1] = 1
    else:
        y = np.zeros((x.shape[0], 1))
        y[0:N,0] = 1


    return torch.from_numpy(x).float(), torch.from_numpy(y).float()

# MNIST and Flipped
def MNIST_data(class_filer=None, aug = True, batch_size=64, shuffle=True, Plot=True):

    # Download MNIST
    trans = transforms.Compose([transforms.ToTensor(),
                           ])

    train_set = torch_datasets.MNIST(root='./data', train=True, transform=trans, download=True)
    test_set = torch_datasets.MNIST(root='./data', train=False, transform=trans, download=True)

    # Filter dataset by class
    if class_filer is not None:
        filters_idxs = train_set.targets == class_filer
        train_set.data = train_set.data[filters_idxs]
        train_set.targets = train_set.targets[filters_idxs]

        filters_idxs = test_set.targets == class_filer
        test_set.data = test_set.data[filters_idxs]
        test_set.targets = test_set.targets[filters_idxs]

    if aug:
        # Add flipped images
        train_set.data = torch.cat((train_set.data, torch.flip(train_set.data, dims=[2])), dim=0)
        train_set.targets = torch.cat((train_set.targets, train_set.targets), dim=0)

        test_set.data = torch.cat((test_set.data, torch.flip(test_set.data, dims=[2])), dim=0)
        test_set.targets = torch.cat((test_set.targets, test_set.targets), dim=0)

        if Plot:
            plot_images(train_set.data, n_images=10, image_dim=28,title='Train Examples')
            plot_images(test_set.data, n_images=10, image_dim=28, title='Test Examples')

        # Create Dataloaders
        train_loader = torch.utils.data.DataLoader(dataset=train_set, batch_size=batch_size, shuffle=shuffle)
        test_loader = torch.utils.data.DataLoader(dataset=test_set, batch_size=batch_size, shuffle=shuffle)

    else:
        train_set = torch_latent_dataloader(train_set.data.numpy(),torch.flip(train_set.data, dims=[2]).numpy(),transform=trans)
        test_set = torch_latent_dataloader(test_set.data.numpy(),torch.flip(test_set.data, dims=[2]).numpy(),transform=trans)

        if Plot:
            
            plot_images(train_set.data, indices = [*range(10)], n_images=10, image_dim=28,title='Train Examples')
            plot_images(train_set.flipped_data, indices = [*range(10)], n_images=10, image_dim=28,title='Train Flipped Examples')

        # Create Dataloaders
        train_loader = torch.utils.data.DataLoader(dataset=train_set, batch_size=batch_size, shuffle=False)
        test_loader = torch.utils.data.DataLoader(dataset=test_set, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader




#### Loss Utils ####

# Define KL loss function for latent space
def vae_KLD(mu,logvar):
    KLD = 0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return - KLD

# Binary Cross Entropy loss function for images
def vae_recon_loss(true, reconstruction, loss = 'bce', image_size = 32,channel = 1):
    """Loss for Variational AutoEncoder Reconstruction"""
    if loss == 'bce':
        return F.binary_cross_entropy(input=reconstruction.view(-1, channel * image_size*image_size), target=true.view(-1, channel * image_size*image_size), reduction='sum')
    elif loss == 'mse':
        return F.mse_loss(input=reconstruction.view(-1, channel * image_size*image_size), target=true.view(-1, channel * image_size*image_size), reduction='sum')

#### Train and Test Utils ####
# Test Function
def test(model, test_loader, device,image_size=32):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for idx, (x,label) in enumerate(test_loader):
            # x = x.reshape(x.shape[0],1,x.shape[1],x.shape[2]).to(device)
            x = x.to(device)
            recon_batch, mu, logvar = model(x)
            test_loss += vae_recon_loss(x, recon_batch,image_size=image_size).item()

        test_loss /= len(test_loader)
    return test_loss

# Test Latent Transform
def test_flip(model, test_loader, device,image_size=32):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for idx, (x,x_flip) in enumerate(test_loader):
            # x = x.reshape(x.shape[0],1,x.shape[1],x.shape[2]).to(device).type(torch.float32)
            # x_flip = x_flip.reshape(x_flip.shape[0],1,x_flip.shape[1],x_flip.shape[2]).to(device).type(torch.float32)
            x = x.to(device)
            x_flip = x_flip.to(device)
            _,z,_ = model(x)
            x_flip_transform = model.decoder(z)
            x_flip_hat,z_flip,_ = model(x_flip)

            test_loss += vae_recon_loss(x_flip_transform, x_flip_hat,image_size=image_size).item() # sum up batch loss

    test_loss /= len(test_loader.dataset)
    return test_loss


#### Plot Utils ####

# Uniform Plot of Images in dataset
def plot_images(images, indices=None, n_images=10, image_dim=32, title=None):

    # Get 10 random indices from length of images
    if indices == None:
        indices = np.random.randint(0, len(images), n_images)

    fig, axes = plt.subplots(1, len(indices), figsize=(20, 2))
    plt.suptitle(title, fontsize=16,fontweight='bold')
    for i, ax in enumerate(axes):
        ax.imshow(images[indices[i]].reshape(image_dim,image_dim), cmap='gray')
        ax.set_title(indices[i])
        ax.axis('off')