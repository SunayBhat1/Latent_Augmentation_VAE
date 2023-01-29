import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.datasets as torch_datasets

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.decomposition import FastICA

import imgaug.augmenters as iaa

from models import torch_dataloader, torch_latent_dataloader
# from IPython import display
# import time
from moviepy.editor import VideoClip
from moviepy.video.io.bindings import mplfig_to_npimage


plt.style.use('seaborn')

##### Data Utils ####################################

# Set Seeds

def project_seed(seed):
    torch.manual_seed(0)
    np.random.seed(0)

# MNIST and Flipped
def MNIST_data(class_filter=None, augs = [], combine = True, batch_size=64, Plot=False, targets = False):
    '''
    Description: Load MNIST dataset with augmentations for LAVAE test and training
    class_filter: filter dataset by a single class or None
    augs: list of augmentations to apply to dataset. Supported augmentations:
        'flip_ud': flip image up and down
        'flip_lr': flip image left and right
        'rotate_cw': rotate image clockwise
        'rotate_ccw': rotate image counter clockwise
        'gaussian_blur': gaussian blur image (kernelsize=7, sigma=(0.1, 2.0))
        'edge_detect': edge detect image (alpha=(0.0, 1))
        'shearX': shear image in x direction between -50:-45 degrees
        'canny': canny edge detection (alpha=1, flip colors)
        'mini_image': add 8x8 version of image in top left corner
        'tl_sap_noise': top left noise
        'sap_noise': noise
    combine: combine original and augmented dataset or leave separate in dictionary
    batch_size: batch size for dataloader
    Plot: plot dataset to visualize augmentations
    '''

    # Download MNIST
    trans = transforms.Compose([transforms.ToTensor()])

    train_set = torch_datasets.MNIST(root='./data', train=True, transform=trans, download=True)
    test_set = torch_datasets.MNIST(root='./data', train=False, transform=trans, download=True)

    # Filter dataset by class
    if class_filter is not None:
        filters_idxs = train_set.targets == class_filter
        train_set.data = train_set.data[filters_idxs]
        train_set.targets = train_set.targets[filters_idxs]

        filters_idxs = test_set.targets == class_filter
        test_set.data = test_set.data[filters_idxs]
        test_set.targets = test_set.targets[filters_idxs]

    train_data = train_set.data
    test_data = test_set.data

    for a in augs: 

        train_list = [train_data]
        test_list = [test_data]

        if a == 'flip_ud': 
            train_list.append(torch.flip(train_data, dims=[1]))
            test_list.append(torch.flip(test_data, dims=[1]))

        elif a == 'flip_lr':
            train_list.append(torch.flip(train_data, dims=[2]))
            test_list.append(torch.flip(test_data, dims=[2]))

        elif a == 'rotate_cw':
            train_list.append(torch.rot90(train_data, k=-1, dims=[1,2]))
            test_list.append(torch.rot90(test_data, k=-1, dims=[1,2]))

        elif a == 'rotate_ccw':
            train_list.append(torch.rot90(train_data, k=1, dims=[1,2]))
            test_list.append(torch.rot90(test_data, k=1, dims=[1,2]))

        elif a == 'gaussian_blur':
            aug = transforms.GaussianBlur(7, sigma=(0.1, 2.0))

            train_list.append(aug(train_data))
            test_list.append(aug(test_data))

        elif a == 'edge_detect':
            aug = iaa.EdgeDetect(alpha=(0.0, 1))

            train_list.append(torch.from_numpy(aug.augment_images(train_data.numpy())))
            test_list.append(torch.from_numpy(aug.augment_images(test_data.numpy())))

        elif a == 'shearX':
            aug = iaa.ShearX((-50, -45))

            train_list.append(torch.from_numpy(aug.augment_images(train_data.numpy())))
            test_list.append(torch.from_numpy(aug.augment_images(test_data.numpy())))

        elif a == 'canny':
            aug = iaa.Canny(alpha=1,    colorizer=iaa.RandomColorsBinaryImageColorizer(
                                color_true=0,
                                color_false=255
                            ))

            train_list.append(torch.from_numpy(aug.augment_images(train_data.numpy())))
            test_list.append(torch.from_numpy(aug.augment_images(test_data.numpy())))

        elif a == 'mini_image':
            aug = transforms.Resize(8)

            copy = train_data.clone()
            copy[:,0:8,0:8] = aug(train_data)
            train_list.append(copy)

            copy = test_data.clone()
            copy[:,0:8,0:8] = aug(test_data)
            test_list.append(copy)

        elif a == 'tl_sap_noise': 
            noisy_train = train_data.clone()
            noise = (128 + 32 * torch.randn((noisy_train.shape[0], noisy_train.shape[1] // 4, noisy_train.shape[2] // 4))).to(torch.uint8)
            noisy_train[:, :noisy_train.shape[1] // 4, :noisy_train.shape[2] // 4] += noise
            train_list.append(noisy_train)

            noisy_test = test_data.clone()
            noise = (128 + 32 * torch.randn((noisy_test.shape[0], 
                                                noisy_test.shape[1] // 4, 
                                                noisy_test.shape[2] // 4))).to(torch.uint8)
            noisy_test[:, :noisy_test.shape[1] // 4, :noisy_test.shape[2] // 4] += noise
            test_list.append(noisy_test)

        elif a == 'sap_noise': 
            noisy_train = train_data.clone()
            noisy_mask = torch.rand(noisy_train.size()) < 0.2
            noise = (128 + 16 * torch.randn(noisy_train.size())).to(torch.int8)
            noisy_train[noisy_mask] += noise[noisy_mask]
            train_list.append(noisy_train)

            noisy_test = test_data.clone()
            noisy_mask = torch.rand(noisy_test.size()) < 0.2
            noise = (128 + 16 * torch.randn(noisy_test.size())).to(torch.int8)
            noisy_test[noisy_mask] += noise[noisy_mask]
            test_list.append(noisy_test)

        else: 
            raise ValueError('augmentation currently not supported')

        train_data = torch.cat(train_list, dim=0)
        test_data = torch.cat(test_list, dim=0)

    # Make aug labels
    if targets:
        train_targets = train_set.targets
        test_targets = test_set.targets

        train_targets = torch.cat([torch.ones_like(train_targets) * i for i in range(len(augs)*2)], dim=0)
        test_targets = torch.cat([torch.ones_like(test_targets) * i for i in range(len(augs)*2)], dim=0)


    # Create Dataloaders
    if combine:
        train_data = torch.cat(train_list, dim=0)
        test_data = torch.cat(test_list, dim=0)

        if targets:
            train_set = torch_dataloader(train_data.numpy(),transform=trans,targets=train_targets)
            test_set = torch_dataloader(test_data.numpy(),transform=trans, targets=test_targets)
        else:
            train_set = torch_dataloader(train_data.numpy(),transform=trans)
            test_set = torch_dataloader(test_data.numpy(),transform=trans)

        train_loader = torch.utils.data.DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True)
        test_loader = torch.utils.data.DataLoader(dataset=test_set, batch_size=batch_size, shuffle=True)

        if Plot:
            plot_images(train_set.data, n_images=10, image_dim=28,title='Train Examples')
            plot_images(test_set.data, n_images=10, image_dim=28, title='Test Examples')

    else:
        if len(augs) == 1:
            
            train_arrays = {'orig': train_list[0].numpy(),
                            augs[0]: train_list[1].numpy()}
            test_arrays = {'orig': test_list[0].numpy(),
                            augs[0]: test_list[1].numpy()}

        elif len(augs) == 2:

            train_list_1 = torch.split(train_list[0], train_list[0].shape[0]//2)
            train_list_2 = torch.split(train_list[1], train_list[1].shape[0]//2)

            train_arrays = {'orig': train_list_1[0].numpy(),
                            augs[0]: train_list_1[1].numpy(),
                            augs[1]: train_list_2[0].numpy(),
                            'compose': train_list_2[1].numpy()}

            test_list_1 = torch.split(test_list[0], test_list[0].shape[0]//2)
            test_list_2 = torch.split(test_list[1], test_list[1].shape[0]//2)

            test_arrays = {'orig': test_list_1[0].numpy(),
                            augs[0]: test_list_1[1].numpy(),
                            augs[1]: test_list_2[0].numpy(),
                            'compose': test_list_2[1].numpy()}

        train_set = torch_latent_dataloader(train_arrays,transform=trans)
        test_set = torch_latent_dataloader(test_arrays,transform=trans)

        train_loader = torch.utils.data.DataLoader(dataset=train_set, batch_size=batch_size, shuffle=False)
        test_loader = torch.utils.data.DataLoader(dataset=test_set, batch_size=batch_size, shuffle=False)


        if Plot:
            for a,data in train_set.data_arrays.items(): 
                plot_images(data, indices = np.arange(10), n_images=10, image_dim=28,title='Train Examples, Aug {}'.format(a))

    return train_loader, test_loader


####  Plotting  ########################

# Uniform Plot of Images in dataset
def plot_images(images, indices=None, n_images=10, image_dim=32, title=None):

    # Get 10 random indices from length of images
    if indices is None:
        indices = np.random.randint(0, len(images), n_images)

    fig, axes = plt.subplots(1, len(indices), figsize=(20, 2))
    plt.suptitle(title, fontsize=16,fontweight='bold')
    for i, ax in enumerate(axes):
        ax.imshow(images[indices[i]].reshape(image_dim,image_dim), cmap='gray')
        ax.set_title(indices[i])
        ax.axis('off')


def plot_reconstructions(model, test_loader, device, num_samples=10, save_path=None):

    plt.style.use('default')

    model.eval()

    # Plotting
    fig, ax = plt.subplots(2, num_samples, figsize=(num_samples, 2))

    # Sample from test set
    for i, x in enumerate(test_loader):
        if i == num_samples:
            break
        x = x.to(device)
        x_recon = model(x)[0]
        index = np.random.randint(0, len(x))  # Random index
        ax[0, i].imshow(x[index].cpu().detach().numpy().squeeze(), cmap='gray')
        ax[1, i].imshow(x_recon[index].cpu().detach().numpy().squeeze(), cmap='gray')
        ax[0, i].axis('off')
        ax[1, i].axis('off')

    plt.subplots_adjust(wspace=0.05, hspace=0.01)

    if save_path is not None:
        plt.savefig(save_path)

def plot_Laug_Recon(model, test_loader, device, augs, decoder=0, num_samples=10, save_path=None):

    model.eval()

    plt.style.use('default')

    # Plotting
    if len(model.Laugs) == 2:
        fig, ax = plt.subplots(8, num_samples, figsize=(num_samples, 8))
    else:
        fig, ax = plt.subplots(3, num_samples, figsize=(num_samples, 3))

    x_laugs = {}
    x_hat = {}
    z = {}

    x = next(iter(test_loader))

    for aug in x.keys(): 
        x[aug] = x[aug].to(device)
        x_hat[aug], z[aug], _ = model(x[aug])

    x_laugs['orig'] = x_hat['orig']
    x_laugs = {aug: model.decoders[decoder](z['orig'] @ model.Laugs[i]) for i,aug in enumerate(augs)}
    if len(model.Laugs) == 2:
        x_laugs['compose'] = model.decoders[decoder](z['orig']@ model.Laugs[0] @ model.Laugs[1])
        x_laugs['r_compose'] = model.decoders[decoder](z['orig']@ model.Laugs[1] @ model.Laugs[0])

    rotation = 0
    if len(model.Laugs) == 2:
        for i in range(num_samples):
            ax[0, i].imshow(x['orig'][i].cpu().detach().numpy().squeeze(), cmap='gray')
            ax[1, i].imshow(x[augs[0]][i].cpu().detach().numpy().squeeze(), cmap='gray')
            ax[2, i].imshow(x_laugs[augs[0]][i].cpu().detach().numpy().squeeze(), cmap='gray')
            ax[3, i].imshow(x[augs[1]][i].cpu().detach().numpy().squeeze(), cmap='gray')
            ax[4, i].imshow(x_laugs[augs[1]][i].cpu().detach().numpy().squeeze(), cmap='gray')
            ax[5, i].imshow(x['compose'][i].cpu().detach().numpy().squeeze(), cmap='gray')
            ax[6, i].imshow(x_laugs['compose'][i].cpu().detach().numpy().squeeze(), cmap='gray')
            ax[7, i].imshow(x_laugs['r_compose'][i].cpu().detach().numpy().squeeze(), cmap='gray')

            if i == 0:
                ax[0, i].set_ylabel('Original Image',rotation=rotation)
                ax[1, i].set_ylabel('Image {}'.format(augs[0]),rotation=rotation)
                ax[2, i].set_ylabel('Latent {}'.format(augs[0]),rotation=rotation)
                ax[3, i].set_ylabel('Image {}'.format(augs[1]),rotation=rotation)
                ax[4, i].set_ylabel('Latent {}'.format(augs[1]),rotation=rotation)
                ax[5, i].set_ylabel('Image Compose',rotation=rotation)
                ax[6, i].set_ylabel('Latent Compose',rotation=rotation)
                ax[7, i].set_ylabel('Latent Reverse\nCompose',rotation=rotation)

            for j in range(8):
                if i == 0: ax[j, i].yaxis.set_label_coords(-1,0.5)
                ax[j, i].set_xticks([])
                ax[j, i].set_yticks([])
    else:
        for i in range(num_samples):
            ax[0, i].imshow(x['orig'][i].cpu().detach().numpy().squeeze(), cmap='gray')
            ax[1, i].imshow(x[augs[0]][i].cpu().detach().numpy().squeeze(), cmap='gray')
            ax[2, i].imshow(x_laugs[augs[1]][i].cpu().detach().numpy().squeeze(), cmap='gray')

            if i == 0:
                ax[0, i].set_ylabel('Original Image',rotation=rotation)
                ax[1, i].set_ylabel('Image Augment',rotation=rotation)
                ax[2, i].set_ylabel('Latent Augment',rotation=rotation)

            for j in range(3):
                if i == 0: ax[j, i].yaxis.set_label_coords(-1,0.5)
                ax[j, i].set_xticks([])
                ax[j, i].set_yticks([])


    plt.subplots_adjust(wspace=0, hspace=0)

    if save_path is not None:
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0)

def plot_Laug_Inv(model, test_loader, device, augs, decoder=0, num_samples=10, save_path=None):

    model.eval()

    plt.style.use('default')

    # Plotting
    fig, ax = plt.subplots(5, num_samples, figsize=(num_samples, 5))

    x_laugs = {}
    x_hat = {}
    z = {}

    x = next(iter(test_loader))

    for aug in x.keys(): 
        x[aug] = x[aug].to(device)
        x_hat[aug], z[aug], _ = model(x[aug])

    x_laugs['orig'] = x_hat['orig']
    x_laugs = {aug: model.decoders[decoder](z['orig'] @ model.Laugs[i] @ model.Laugs[i]) for i,aug in enumerate(augs)}
    x_laugs['compose'] = model.decoders[decoder](z['orig']@ model.Laugs[0] @ model.Laugs[1] @ model.Laugs[0] @ model.Laugs[1])
    x_laugs['r_compose'] = model.decoders[decoder](z['orig']@ model.Laugs[1] @ model.Laugs[0]@ model.Laugs[1] @ model.Laugs[0])

    rotation = 0
    for i in range(num_samples):
        ax[0, i].imshow(x['orig'][i].cpu().detach().numpy().squeeze(), cmap='gray')
        ax[1, i].imshow(x_laugs[augs[0]][i].cpu().detach().numpy().squeeze(), cmap='gray')
        ax[2, i].imshow(x_laugs[augs[1]][i].cpu().detach().numpy().squeeze(), cmap='gray')
        ax[3, i].imshow(x_laugs['compose'][i].cpu().detach().numpy().squeeze(), cmap='gray')
        ax[4, i].imshow(x_laugs['r_compose'][i].cpu().detach().numpy().squeeze(), cmap='gray')

        if i == 0:
            ax[0, i].set_ylabel('Original Image',rotation=rotation)
            ax[1, i].set_ylabel('Latent {}\nInverse'.format(augs[0]),rotation=rotation)
            ax[2, i].set_ylabel('Latent {}\nInverse'.format(augs[1]),rotation=rotation)
            ax[3, i].set_ylabel('Compose\nInverse'.format(augs[0]),rotation=rotation)
            ax[4, i].set_ylabel('Reverse Compose\nInverse'.format(augs[1]),rotation=rotation)

        for j in range(5):
            if i == 0: ax[j, i].yaxis.set_label_coords(-1,0.5)
            ax[j, i].set_xticks([])
            ax[j, i].set_yticks([])


    plt.subplots_adjust(wspace=0, hspace=0)

    if save_path is not None:
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0)


def make_2D_video(model, test_loader, device, augs, decoder=0, save_path=None, types = ['TSNE']):

    plt.style.use('seaborn')

    model.eval()

    # Sample from test set
    images = []
    z_s = []
    aug_labels = []

    latents = []
    latent_recons = []
    latent_aug_labels = []

    for i, (x,lab) in enumerate(test_loader):
        x = x.to(device)
        x_hat,z,_ = model(x)

        images.append(x.cpu().detach().numpy())
        z_s.append(z.cpu().detach().numpy())
        # recons.append(x_hat.cpu().detach().numpy())
        aug_labels.append(lab.cpu().detach().numpy())

    images = np.concatenate(images)
    z_s = np.concatenate(z_s)
    # recons = np.concatenate(recons)
    aug_labels = np.concatenate(aug_labels)

    # Latent reconstruction
    latent_recons.append(images[aug_labels==0])
    latents.append(z_s[aug_labels==0])
    latent_aug_labels.append(torch.ones(images[aug_labels==0].shape[0]).cpu().detach().numpy() * 0)

    for i,aug in enumerate(augs):
        z = torch.from_numpy(z_s[aug_labels==0]) @ model.Laugs[i]
        latents.append(z.cpu().detach().numpy())
        x_latent = model.decoders[decoder](z)
        latent_recons.append(x_latent.cpu().detach().numpy())
        latent_aug_labels.append(torch.ones(x_latent.shape[0]).cpu().detach().numpy() * (i+1))

    if len(model.Laugs) == 2:
        z = torch.from_numpy(z_s[aug_labels==0]) @ model.Laugs[0] @ model.Laugs[1]
        latents.append(z.cpu().detach().numpy())
        x_latent = model.decoders[decoder](z)
        latent_recons.append(x_latent.cpu().detach().numpy())
        latent_aug_labels.append(torch.ones(x_latent.shape[0]).cpu().detach().numpy() * 3)

        z = torch.from_numpy(z_s[aug_labels==0]) @ model.Laugs[1] @ model.Laugs[0]
        latents.append(z.cpu().detach().numpy())
        x_latent = model.decoders[decoder](z)
        latent_recons.append(x_latent.cpu().detach().numpy())
        latent_aug_labels.append(torch.ones(x_latent.shape[0]).cpu().detach().numpy() * 4)

    latents = np.concatenate(latents)
    latent_recons = np.concatenate(latent_recons)
    latent_aug_labels = np.concatenate(latent_aug_labels)

    # 2-D Projections
    pca = PCA(n_components=2)
    tsne = TSNE(n_components=2, perplexity=40, n_iter=300)
    ica = FastICA(n_components=2, max_iter=1000)

    project = {'PCA':{}, 'TSNE':{}, 'ICA':{}}

    item_labels = ['Images', 'Latent','Recons']
    for i,item in enumerate([images, latents, latent_recons]):
        project['PCA'][item_labels[i]] = pca.fit_transform(item.reshape(item.shape[0],-1))
        project['TSNE'][item_labels[i]] = tsne.fit_transform(item.reshape(item.shape[0],-1))
        project['ICA'][item_labels[i]] = ica.fit_transform(item.reshape(item.shape[0],-1) + 1e-6)

    iimg = 0
    labels = ['Orig',augs[0], augs[1],'Compose']
    fig, axes = plt.subplots(len(types), 3, figsize=(15,5))
    for num_samples in range(100):
        for i in range(5):
            for j,type in enumerate(types):
                for k,item in enumerate(item_labels):
                    if k ==0:
                        if i == 4: continue
                        if len(types) ==  1: ax = axes[k]
                        else: ax = axes[j][k]
                        scatters = project[type][item][aug_labels==i]
                        index = np.random.choice(scatters.shape[0], 1)
                        ax.scatter(scatters[index,0], scatters[index,1],label=labels[i], alpha=0.5,c=plt.rcParams['axes.prop_cycle'].by_key()['color'][i])

                    else:
                        if i == 4: 
                            alpha = 0.1
                            label = 'Reverse\nCompose'
                        else: 
                            alpha = 0.5
                            label = labels[i]
                        if len(types) ==  1: ax = axes[k]
                        else: ax = axes[j][k]
                        scatters = project[type][item][latent_aug_labels==i]
                        index = np.random.choice(scatters.shape[0], 1)
                        ax.scatter(scatters[index,0], scatters[index,1],label=label, alpha=alpha,c=plt.rcParams['axes.prop_cycle'].by_key()['color'][i])
                        mplfig_to_npimage(fig)
                    if len(types) ==  1: ax = axes[k]
                    else: ax = axes[j][k]
                    ax.set_title('{} {}'.format(type, item))
                    fig.suptitle('2-D Projection Scatter Plots', fontsize=20, fontweight='bold')

                    if save_path is not None:
                        plt.savefig(save_path + 'file%06d.png' % iimg, bbox_inches='tight', pad_inches=0)
                        iimg += 1

def plot_2D_spaces(model, test_loader, device, augs, decoder=0, save_path=None, types = ['PCA', 'TSNE', 'ICA']):

    plt.style.use('seaborn')

    model.eval()

    # Sample from test set
    images = []
    z_s = []
    aug_labels = []

    latents = []
    latent_recons = []
    latent_aug_labels = []

    for i, (x,lab) in enumerate(test_loader):
        x = x.to(device)
        x_hat,z,_ = model(x)

        images.append(x.cpu().detach().numpy())
        z_s.append(z.cpu().detach().numpy())
        # recons.append(x_hat.cpu().detach().numpy())
        aug_labels.append(lab.cpu().detach().numpy())

    images = np.concatenate(images)
    z_s = np.concatenate(z_s)
    # recons = np.concatenate(recons)
    aug_labels = np.concatenate(aug_labels)

    # Latent reconstruction
    latent_recons.append(images[aug_labels==0])
    latents.append(z_s[aug_labels==0])
    latent_aug_labels.append(torch.ones(images[aug_labels==0].shape[0]).cpu().detach().numpy() * 0)

    for i,aug in enumerate(augs):
        z = torch.from_numpy(z_s[aug_labels==0]).to(device) @ model.Laugs[i]
        latents.append(z.cpu().detach().numpy())
        x_latent = model.decoders[decoder](z)
        latent_recons.append(x_latent.cpu().detach().numpy())
        latent_aug_labels.append(torch.ones(x_latent.shape[0]).cpu().detach().numpy() * (i+1))

    if len(model.Laugs) == 2:
        z = torch.from_numpy(z_s[aug_labels==0]).to(device) @ model.Laugs[0] @ model.Laugs[1]
        latents.append(z.cpu().detach().numpy())
        x_latent = model.decoders[decoder](z)
        latent_recons.append(x_latent.cpu().detach().numpy())
        latent_aug_labels.append(torch.ones(x_latent.shape[0]).cpu().detach().numpy() * 3)

        z = torch.from_numpy(z_s[aug_labels==0]).to(device) @ model.Laugs[1] @ model.Laugs[0]
        latents.append(z.cpu().detach().numpy())
        x_latent = model.decoders[decoder](z)
        latent_recons.append(x_latent.cpu().detach().numpy())
        latent_aug_labels.append(torch.ones(x_latent.shape[0]).cpu().detach().numpy() * 4)

    for i in range(len(latents)):
        print(latents[i].shape)
    latents = np.concatenate(latents)
    latent_recons = np.concatenate(latent_recons)
    latent_aug_labels = np.concatenate(latent_aug_labels)

    # 2-D Projections
    pca = PCA(n_components=2)
    tsne = TSNE(n_components=2, perplexity=40, n_iter=300)
    ica = FastICA(n_components=2, max_iter=1000)

    project = {'PCA':{}, 'TSNE':{}, 'ICA':{}}

    item_labels = ['Images', 'Latent','Recons']
    for i,item in enumerate([images, latents, latent_recons]):
        project['PCA'][item_labels[i]] = pca.fit_transform(item.reshape(item.shape[0],-1))
        project['TSNE'][item_labels[i]] = tsne.fit_transform(item.reshape(item.shape[0],-1))
        project['ICA'][item_labels[i]] = ica.fit_transform(item.reshape(item.shape[0],-1) + 1e-6)

    labels = ['Orig',augs[0], augs[1],'Compose']
    fig, axes = plt.subplots(len(types), 3, figsize=(15,10))
    for i in range(5):
        for j,type in enumerate(types):
            for k,item in enumerate(item_labels):
                if k ==0:
                    if i == 4: continue
                    axes[j][k].scatter(project[type][item][aug_labels==i,0], project[type][item][aug_labels==i,1],label=labels[i], alpha=0.5)
                else:
                    if i == 4: 
                        alpha = 0.1
                        label = 'Reverse\nCompose'
                    else: 
                        alpha = 0.5
                        label = labels[i]
                    axes[j][k].scatter(project[type][item][latent_aug_labels==i,0], project[type][item][latent_aug_labels==i,1],label=label, alpha=alpha)
                axes[j][k].set_title('{} {}'.format(type, item))

    fig.suptitle('2-D Projection Scatter Plots', fontsize=20, fontweight='bold')
    axes[0][2].legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)

    if save_path is not None:
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
        

def plot_2D_spaces_Inv(model, test_loader, device, augs, decoder=0, save_path=None, types = ['PCA', 'TSNE', 'ICA']):

    plt.style.use('seaborn')

    model.eval()

    # Sample from test set
    images = []
    z_s = []
    aug_labels = []

    latents = []
    latent_recons = []
    latent_aug_labels = []

    for i, (x,lab) in enumerate(test_loader):
        x = x.to(device)
        x_hat,z,_ = model(x)

        images.append(x.cpu().detach().numpy())
        z_s.append(z.cpu().detach().numpy())
        # recons.append(x_hat.cpu().detach().numpy())
        aug_labels.append(lab.cpu().detach().numpy())

    images = np.concatenate(images)
    z_s = np.concatenate(z_s)
    # recons = np.concatenate(recons)
    aug_labels = np.concatenate(aug_labels)

    # Latent reconstruction
    latent_recons.append(images[aug_labels==0])
    latents.append(z_s[aug_labels==0])
    latent_aug_labels.append(torch.ones(images[aug_labels==0].shape[0]).cpu().detach().numpy() * 0)

    for i,aug in enumerate(augs):
        z = torch.from_numpy(z_s[aug_labels==0]).to(device) @ model.Laugs[i] @ model.Laugs[i]
        latents.append(z.cpu().detach().numpy())
        x_latent = model.decoders[decoder](z)
        latent_recons.append(x_latent.cpu().detach().numpy())
        latent_aug_labels.append(torch.ones(x_latent.shape[0]).cpu().detach().numpy() * (i+1))

    latents = np.concatenate(latents)
    latent_recons = np.concatenate(latent_recons)
    latent_aug_labels = np.concatenate(latent_aug_labels)

    # 2-D Projections
    pca = PCA(n_components=2)
    tsne = TSNE(n_components=2, perplexity=40, n_iter=300)
    ica = FastICA(n_components=2, max_iter=1000)

    project = {'PCA':{}, 'TSNE':{}, 'ICA':{}}

    item_labels = ['Images', 'Latent','Recons']
    for i,item in enumerate([images, latents, latent_recons]):
        project['PCA'][item_labels[i]] = pca.fit_transform(item.reshape(item.shape[0],-1))
        project['TSNE'][item_labels[i]] = tsne.fit_transform(item.reshape(item.shape[0],-1))
        project['ICA'][item_labels[i]] = ica.fit_transform(item.reshape(item.shape[0],-1) + 1e-6)

    labels = ['Orig',augs[0]+'_Inv', augs[1]+'_Inv']
    fig, axes = plt.subplots(len(types), 3, figsize=(15,10))
    for i in range(3):
        for j,type in enumerate(types):
            for k,item in enumerate(item_labels):
                if k == 0:
                    axes[j][k].scatter(project[type][item][aug_labels==i,0], project[type][item][aug_labels==i,1],label=labels[i], alpha=0.5)
                else:
                    axes[j][k].scatter(project[type][item][latent_aug_labels==i,0], project[type][item][latent_aug_labels==i,1],label=labels[i], alpha=0.1)
                
                axes[j][k].set_title('{} {}'.format(type, item))

    fig.suptitle('2-D Projection Scatter Plots/nInverses', fontsize=20, fontweight='bold')
    axes[0][2].legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)

    if save_path is not None:
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0)



def plot_2D(model, test_loader, device, augs, decoder=0, save_path=None):


    plt.style.use('seaborn')

    model.eval()

    # Sample from test set
    images = []
    z_s = []
    aug_labels = []

    latents = []
    latent_recons = []
    latent_aug_labels = []

    for i, (x,lab) in enumerate(test_loader):
        x = x.to(device)
        x_hat,z,_ = model(x)

        images.append(x.cpu().detach().numpy())
        z_s.append(z.cpu().detach().numpy())
        # recons.append(x_hat.cpu().detach().numpy())
        aug_labels.append(lab.cpu().detach().numpy())

    images = np.concatenate(images)
    z_s = np.concatenate(z_s)
    # recons = np.concatenate(recons)
    aug_labels = np.concatenate(aug_labels)

    # Latent reconstruction
    latent_recons.append(images[aug_labels==0])
    latents.append(z_s[aug_labels==0])
    latent_aug_labels.append(torch.ones(images[aug_labels==0].shape[0]).cpu().detach().numpy() * 0)

    for i,aug in enumerate(augs):
        z = torch.from_numpy(z_s[aug_labels==0]).to(device) @ model.Laugs[i]
        latents.append(z.cpu().detach().numpy())
        x_latent = model.decoders[decoder](z)
        latent_recons.append(x_latent.cpu().detach().numpy())
        latent_aug_labels.append(torch.ones(x_latent.shape[0]).cpu().detach().numpy() * (i+1))

    if len(model.Laugs) == 2:
        z = torch.from_numpy(z_s[aug_labels==0]).to(device) @ model.Laugs[0] @ model.Laugs[1]
        latents.append(z.cpu().detach().numpy())
        x_latent = model.decoders[decoder](z)
        latent_recons.append(x_latent.cpu().detach().numpy())
        latent_aug_labels.append(torch.ones(x_latent.shape[0]).cpu().detach().numpy() * 3)

        z = torch.from_numpy(z_s[aug_labels==0]).to(device) @ model.Laugs[1] @ model.Laugs[0]
        latents.append(z.cpu().detach().numpy())
        x_latent = model.decoders[decoder](z)
        latent_recons.append(x_latent.cpu().detach().numpy())
        latent_aug_labels.append(torch.ones(x_latent.shape[0]).cpu().detach().numpy() * 4)

    for i in range(len(latents)):
        print(latents[i].shape)
    latents = np.concatenate(latents)
    latent_recons = np.concatenate(latent_recons)
    latent_aug_labels = np.concatenate(latent_aug_labels)
    print(latents.shape)
    print(latent_aug_labels.shape)

    labels = ['Orig',augs[0], augs[1],'Compose','Reverse Compose']
    fig, axes = plt.subplots(1, 2, figsize=(10,5))
    for i in range(5):
        if i == 4: alpha = 0.1
        else: alpha = 0.5
        axes[0].scatter(z_s[aug_labels==i,0], z_s[aug_labels==i,1],label=labels[i], alpha=alpha)
        axes[1].scatter(latents[latent_aug_labels==i,0], latents[latent_aug_labels==i,1],label=labels[i], alpha=alpha)

    axes[0].set_title('Augmented Image (z)')
    axes[1].set_title('Latent Augmentation (Laug(z)')
    fig.suptitle('2-D Projection Scatter Plots', fontsize=20, fontweight='bold')
    axes[1].legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)

    if save_path is not None:
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0)

def sample_latent(model,train_loader,device,augs,decoder=0,num_samples=12,save_path=None):
    z_s = []
    x_sampled = {}
    for i, (x, y) in enumerate(train_loader):
        x = x.to(device)
        _, z, _ = model(x[y==0])

        z_s.append(z.cpu().detach().numpy())

    z_s = np.concatenate(z_s, axis=0)
    num_samples = 16
    z_sampled = np.random.normal(z_s.mean(axis=0).reshape(-1,1).repeat(num_samples, axis=1), z_s.std(axis=0).reshape(-1,1).repeat(num_samples, axis=1))

    z_sampled = torch.from_numpy(z_sampled).float().to(device)
    x_sampled['orig'] = model.decoders[decoder](z_sampled).cpu().detach().numpy()
    x_sampled[augs[0]] = model.decoders[decoder](z_sampled @ model.Laugs[0]).cpu().detach().numpy()
    x_sampled[augs[1]] = model.decoders[decoder](z_sampled @ model.Laugs[1]).cpu().detach().numpy()
    x_sampled['compose'] = model.decoders[decoder](z_sampled @ model.Laugs[0] @ model.Laugs[1]).cpu().detach().numpy()
    x_sampled['r_compose'] = model.decoders[decoder](z_sampled @ model.Laugs[1] @ model.Laugs[0]).cpu().detach().numpy()

    fig, ax = plt.subplots(1, 5, figsize=(4,12))
    for i, (k, v) in enumerate(x_sampled.items()):
        ax[i].imshow(v.reshape(-1, 28), cmap='gray')
        ax[i].set_title(k,rotation=45)
        ax[i].axis('off')
        ax[i].set_xticks([])
        ax[i].set_yticks([])
    plt.subplots_adjust(wspace=0.1, hspace=0.1)