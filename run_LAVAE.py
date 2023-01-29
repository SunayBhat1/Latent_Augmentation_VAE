import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import argparse

import torch
import torch.utils.data as data
import torch.nn.functional as F

from adabelief_pytorch import AdaBelief

from models import LAVAE, torch_latent_dataloader
import utils as ut
import train_utils as tu

def epoch_ED(model, train_loader, optimizer, losses, device, args):

    losses['KLD'].append(0)
    losses['Recon'].append(0)
    losses['Total'].append(0)

    model.train()
    for idx, x in enumerate(train_loader):

        x = x.to(device)
        optimizer.zero_grad()

        x_hat, mu, logvar = model(x)

        loss_KLD = tu.vae_KLD(mu,logvar)
        loss_recon = tu.vae_recon_loss(x, x_hat,image_size=args.IMG_DIM)
        loss = args.LAMBDA_KLD * loss_KLD + args.LAMBDA_RECON * loss_recon
        loss = loss_KLD + loss_recon
        loss.backward()
        optimizer.step()

        losses['KLD'][-1] += loss_KLD.item()
        losses['Recon'][-1] += loss_recon.item()
        losses['Total'][-1] += loss.item()

    losses['KLD'][-1] /= len(train_loader)
    losses['Recon'][-1] /= len(train_loader)
    losses['Total'][-1] /= len(train_loader)

def epoch_LA(model, train_loader, optimizer, losses, device, args):

        MSE_latent = torch.nn.MSELoss()
    
        losses['MSE'].append(0)
        losses['Invol'].append(0)
        losses['Total'].append(0)
    
        for idx, x in enumerate(train_loader):

            for aug in x.keys(): 
                x[aug] = x[aug].to(device)
    
            with torch.no_grad():
                model.eval()
                z_mu_log = {k: model(v)[1:3] for k,v in x.items()}
                model.training=True
                z = {k: model.latent_sample(z_mu_log[k][0], z_mu_log[k][1]) for k,v in x.items()}
                model.training=False
    
            model.train()
            optimizer.zero_grad()

            z_hat = {aug: z['orig'] @ model.Laugs[i]  for i,aug in enumerate(args.AUGS)}

            mse_loss = sum([MSE_latent(z_hat[aug], z[aug]) for aug in args.AUGS])
            invol_loss = tu.invol_loss(model,device)
            loss = mse_loss + args.LAMBDA_INVOL * invol_loss

            loss.backward()
            optimizer.step()
    
            losses['MSE'][-1] += mse_loss.item()
            losses['Invol'][-1] += invol_loss.item()
            losses['Total'][-1] += loss.item()
    
        losses['MSE'][-1] /= len(train_loader)
        losses['Invol'][-1] /= len(train_loader)
        losses['Total'][-1] /= len(train_loader)


def epoch_Decs(model, train_loader, optimizer, losses, device, args, decoder):

    losses['Train'].append(0)

    for idx, x in enumerate(train_loader):

        z = {}
        x_laugs = {}

        for aug in x.keys(): 
            x[aug] = x[aug].to(device)

        with torch.no_grad():
            model.eval()
            
            z['orig'] = model(x['orig'])[1]
            for i,aug in enumerate(args.AUGS_TANSFER[decoder-1]):
                z[aug] = z['orig'] @ model.Laugs[i]
            if len(model.Laugs) == 2:
                z['compose'] = z['orig'] @ model.Laugs[0] @ model.Laugs[1]

        model.train()
        optimizer.zero_grad()

        x_laugs['orig'] = model.decoders[decoder](z['orig'])
        for i,aug in enumerate(args.AUGS_TANSFER[decoder-1]):
            x_laugs[aug] = model.decoders[decoder](z[aug])
        if len(model.Laugs) == 2:
            x_laugs['compose'] = model.decoders[1](z['compose'])

        loss = sum([tu.vae_recon_loss(x[aug], x_laugs[aug],image_size=args.IMG_DIM) for aug in x.keys()])

        loss.backward()
        optimizer.step()

        losses['Train'][-1] += loss.item()

    losses['Train'][-1] /= len(train_loader)

def train_LAVAE(mode, model, train_loader, test_loader, optimizer, device, args,decoder=None):
    '''
    Train LAVAE model
    Inputs:
        mode: 'ED': Encoder/Decoder,'LA': Latent Augs Matrices,'Decs': Transfer Decoders
        model: LAVAE model
        train_loader: Training Dataloader
        test_loader: Testing Dataloader
        optimizer: Optimizer
        device: Device to train on
        args: Argparse arguments
    '''

    # Define Losses
    if mode == 'ED':
        if args.VERBOSE: print('Training LAVAE Encoder, Decoder...')
        losses = {'KLD':[], 
                  'Recon':[],
                  'Total':[],
                  'Test': [],
                 }   
        epochs = args.ED_EPOCHS

    elif mode == 'LA':
        if args.VERBOSE: print('Training LAVAE Latent Aug Matrices...')
        losses = {'MSE':[], 
                  'Invol':[],
                  'Total':[],
                  'Test': [],
                 }   
        epochs = args.LA_EPOCHS
    elif mode == 'Decs':
        if decoder == 1: model.losses[mode] = {}
        if args.VERBOSE: print('Training LAVAE Decoder {}/{}...'.format(decoder+1,args.num_dec))
        losses = {'Train':[],
                  'Test': [],
                 }
        epochs = args.DEC_EPOCHS

    # Training Loop
    for epoch in range(1,epochs+1):
        if mode == 'ED':
            epoch_ED(model, train_loader, optimizer, losses, device, args)
            losses['Test'].append(tu.test_ED(model, test_loader, device, image_size=args.IMG_DIM))
        elif mode == 'LA':
            epoch_LA(model, train_loader, optimizer, losses, device, args)
            losses['Test'].append(tu.test_LA(model, test_loader, device, args))
        elif mode == 'Decs':
            epoch_Decs(model, train_loader, optimizer, losses, device, args, decoder)
            losses['Test'].append(tu.test_Decs(model, test_loader, device, args, args.AUGS_TANSFER[decoder-1], decoder))

        if epoch % args.CHECKPOINTS == 0:
            loss_string = ['{}: {:.4f}'.format(k,v[-1]) for k,v in losses.items()]
            print(f'Epoch {epoch}/{epochs} - ', ', '.join(loss_string))
            
            if mode == 'Decs':
                model.losses[mode][decoder-1] = losses
                torch.save(model, os.path.join(args.checkpoints_folder,'{}_{}_{}.pt'.format(mode,decoder,epoch)))
            else:
                model.losses[mode] = losses
                torch.save(model, os.path.join(args.checkpoints_folder,'{}_{}.pt'.format(mode,epoch)))



def main():

    # Argparse and required arguments
    parser=argparse.ArgumentParser(description='Run Latent Augmentation VAE')
    parser.add_argument('--experiment_name', type=str, help='Name of Experiment')
    
    # Experiment Args
    parser.add_argument('--AUGS', nargs="*", type=str, default=['flip_lr', 'flip_ud'], help='Original Augmentations to train latent space ')
    parser.add_argument('--AUGS_TANSFER', nargs="*", type=str, default=None, help='Augmentations to transfer latent space to (must be even)')
    parser.add_argument('--SEED', type=int, default=0, help='Augmentations to transfer latent space to (must be even)')
    parser.add_argument('--LOAD_MODEL', action='store_true', help='Load model from checkpoints folder')

    # Model Args
    parser.add_argument('--LATENT_DIM', type=int, default=16, help='Dimension of latent space')
    parser.add_argument('--NUM_FILTERS', type=int, default=64, help='Number of filters in convolutional layers')
    parser.add_argument('--LINEAR', action='store_false', help='Use non-linear layers instead of convolutions')

    # Optimizer Args
    parser.add_argument('--OPTIM', type=str, default='Adabelief', help='Optimizer to use (Adabelief)')
    
    # Training Args
    parser.add_argument('--CLASS_FILTER', type=int, default=None, help='Classes to filter dataset to')
    parser.add_argument('--CHECKPOINTS', type=int, default=20, help='Number of epochs to save checkpoint and print losses')
    parser.add_argument('--ED_EPOCHS', type=int, default=100, help='Number of epochs to train encoder/decoder')
    parser.add_argument('--LA_EPOCHS', type=int, default=60, help='Number of epochs to train latent augmentations')
    parser.add_argument('--DEC_EPOCHS', type=int, default=100, help='Number of epochs to train a transfer decoder head')
    parser.add_argument('--DATA', type=str, default='mnist', help='Dataset to use (mnist, sketch, fashion, cifar10)')
    parser.add_argument('--BATCH_SIZE', type=int, default=64, help='Batch size')
    parser.add_argument('--LAMBDA_KLD', type=int, default=5, help='ED KLD weight')
    parser.add_argument('--LAMBDA_RECON', type=int, default=1, help='ED Reconstruction weight')
    parser.add_argument('--LAMBDA_INVOL', type=int, default=1, help='Involutary Matrix loss weight')

    # Run Arguments
    parser.add_argument('--VERBOSE', default=True, action=argparse.BooleanOptionalAction)

    args = parser.parse_args()

    ##### Initialize #####

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Set seed
    if args.SEED is not None:
        ut.project_seed(args.SEED)

    # Results Folder
    args.results_folder = os.path.join('Results', args.experiment_name)
    args.checkpoints_folder = os.path.join(args.results_folder, 'Checkpoints')
    args.plots_folder = os.path.join(args.results_folder, 'Plots')
    print(args.results_folder)
    if not os.path.exists(args.results_folder): 
        os.makedirs(args.results_folder)
        os.makedirs(args.checkpoints_folder)
        os.makedirs(args.plots_folder)

    # Print args to text file and command line if verbose
    with open(os.path.join(args.results_folder, 'args.txt'), 'w') as f:
        for key, value in vars(args).items():
            f.write(f'{key}: {value}\n')

    if args.VERBOSE:
        print('Arguments: \n')
        for key, value in vars(args).items():
            print(f'{key}: {value}')

    # Parse Augmentations for Num Transfers and Decoder Heads
    if args.AUGS_TANSFER is None:
        args.num_dec = 1
        ALL_AUGS = args.AUGS
    else:
        if len(args.AUGS_TANSFER) % 2 != 0:
            raise ValueError('Augmentations to transfer must be even')
        args.num_dec = 1 + int(len(args.AUGS_TANSFER)/2)

        transfer_augs = []
        for i in range(0,len(args.AUGS_TANSFER),2):
            transfer_augs.append([args.AUGS_TANSFER[i], args.AUGS_TANSFER[i+1]])
        args.AUGS_TANSFER = transfer_augs

    if args.VERBOSE:
        print('Number of Decoders: {}'.format(args.num_dec))
        print('Transfer: {}'.format(args.AUGS_TANSFER))

    # Load ED Datasets
    train_loader, test_loader = ut.MNIST_data(class_filter=args.CLASS_FILTER,augs=args.AUGS)
    if args.VERBOSE: print('ED Train/Test Size: {}/{}'.format(len(train_loader.dataset),len(test_loader.dataset)))

    # Image dim
    args.IMG_DIM = train_loader.dataset[0][0].shape[0]
    if args.VERBOSE: print('Image Dim: {}'.format(args.IMG_DIM))

    # Init Model
    if  not args.LOAD_MODEL:
        model = LAVAE(args.LATENT_DIM, args.num_dec, args.NUM_FILTERS, args.IMG_DIM, args.AUGS, args.LINEAR).to(device)
        if args.VERBOSE: print('Model Parameters: {}'.format(sum(p.numel() for p in model.parameters() if p.requires_grad)))

    # Init Optimizer
    if args.OPTIM == 'Adabelief':
        # ED Optimizer
        parameters = list(model.encoder.parameters()) + list(model.decoders[0].parameters())
        optimizer_ED = AdaBelief(parameters,lr=1e-3, eps=1e-16, betas=(0.9,0.999), weight_decouple = False, rectify = False,print_change_log = False)
        # LA Optimizer
        optimizer_LA = AdaBelief(model.Laugs.parameters(),lr=1e-3, eps=1e-16, betas=(0.9,0.999), weight_decouple = False, rectify = False,print_change_log = False)
        # E2 Optimizer
        optimizer_Decs = []
        for dec in range(1,len(model.decoders)):
            optimizer_Decs.append(AdaBelief(model.decoders[dec].parameters(),lr=1e-3, eps=1e-16, betas=(0.9,0.999), weight_decouple = False, rectify = False,print_change_log = False))
    else:
        raise ValueError('Optimizer not supported')

    ##### Train #####

    # Train ED
    if args.LOAD_MODEL:
        model = torch.load(os.path.join(args.checkpoints_folder, 'ED_{}.pt'.format(args.ED_EPOCHS)))
    else:
        train_LAVAE('ED', model, train_loader, test_loader, optimizer_ED, device, args)

    # Plot and Save Reconstruction Results
    ut.plot_reconstructions(model, test_loader, device, num_samples=10, save_path=os.path.join(args.plots_folder, 'ED_Recon.png'))

    # Load LA Datasets
    train_loader, test_loader = ut.MNIST_data(class_filter=args.CLASS_FILTER,augs=args.AUGS, combine=False)
    if args.VERBOSE: print('LA Train/Test Size: {}/{}'.format(len(train_loader.dataset),len(test_loader.dataset)))
    
    # Train or Load LA
    if args.LOAD_MODEL:
        model = torch.load(os.path.join(args.checkpoints_folder, 'LA_{}.pt'.format(args.LA_EPOCHS)))
    else:
        train_LAVAE('LA', model, train_loader, test_loader, optimizer_LA, device, args)
        

    # Plot and Save Latent Augmentation Results
    model.losses['Compose'] = {}
    ut.plot_Laug_Recon(model, test_loader, device, args.AUGS, 0, num_samples=10, save_path=os.path.join(args.plots_folder, 'LA_Recon.png'))
    model.losses['Compose'][0] = tu.test_compose(model, test_loader, device, decoder=0)
    _, test_loader = ut.MNIST_data(class_filter=2,augs=args.AUGS,combine=True,targets=True)
    ut.plot_2D_spaces(model, test_loader, device, args.AUGS, 0, save_path=os.path.join(args.plots_folder, 'LA_2D.png'))


    # Train Transfer Decoders (if needed)
    if args.num_dec > 1:
        
        # Load Transfer Datasets
        for iDec in range(1,args.num_dec):
            train_loader, test_loader = ut.MNIST_data(class_filter=args.CLASS_FILTER,augs=args.AUGS_TANSFER[iDec-1], combine=False)

            # Train or Load Transfer Decoder
            if args.LOAD_MODEL:
                model = torch.load(os.path.join(args.checkpoints_folder, 'Decs_{}_{}.pt'.format(iDec,args.DEC_EPOCHS)))
            else:
                train_LAVAE('Decs', model, train_loader, test_loader, optimizer_Decs[iDec-1], device, args,decoder=iDec)

            # Plot and Save Reconstruction Results
            ut.plot_Laug_Recon(model, test_loader, device, args.AUGS_TANSFER[iDec-1], iDec, num_samples=10, save_path=os.path.join(args.plots_folder, 'Dec_{}_Recon.png'.format(iDec)))
            model.losses['Compose'][iDec] = tu.test_compose(model, test_loader, device, decoder=iDec)
            _, test_loader = ut.MNIST_data(class_filter=2,augs=args.AUGS_TANSFER[iDec-1],combine=True,targets=True)
            ut.plot_2D_spaces(model, test_loader, device, args.AUGS_TANSFER[iDec-1], iDec, save_path=os.path.join(args.plots_folder, 'Dec_{}_2D.png'.format(iDec)))


# Init main
if __name__ == '__main__':
    main()
