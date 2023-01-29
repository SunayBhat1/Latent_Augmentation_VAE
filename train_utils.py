import torch
import torch.nn.functional as F



#### Losses ####################

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

# Involutary Matrix Loss
def invol_loss(model, device, loss = 'mse'):
    # (I - Laug^2).sum()
    I = torch.eye(model.Laugs[0].shape[0]).to(device)
    loss = 0

    for aug in range(len(model.Laugs)):
        loss += torch.square(I - model.Laugs[aug]@model.Laugs[aug]).sum()

    return loss

#### Test #######################

# Test Function
def test_ED(model, test_loader, device,image_size=32):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for idx, x in enumerate(test_loader):
            x = x.to(device)
            recon_batch, mu, logvar = model(x)
            test_loss += vae_recon_loss(x, recon_batch,image_size=image_size).item()
        test_loss /= len(test_loader)
    return test_loss

# Test Latent Transform
def test_LA(model, test_loader, device, args, decoder=0):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for idx, x in enumerate(test_loader):
            x_hat = {}
            z = {}
            for aug in x.keys(): 
                x[aug] = x[aug].to(device)
                x_hat[aug], z[aug], _ = model(x[aug])
            x_laugs = {aug: model.decoders[decoder](z['orig'] @ model.Laugs[i]) for i,aug in enumerate(model.trained_augs)}
            
            test_loss += sum([vae_recon_loss(x_laugs[aug], x[aug], image_size=args.IMG_DIM) for aug in model.trained_augs])

    test_loss /= len(test_loader.dataset)
    return test_loss

# Test Secondary (Transfer) Decoder Heads
def test_Decs(model, test_loader, device, args, augs, decoder=1):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for idx, imgs in enumerate(test_loader):

            x_laugs = {}
            x_hat = {}
            z = {}

            for aug in imgs.keys(): 
                imgs[aug] = imgs[aug].to(device)
                x_hat[aug], z[aug], _ = model(imgs[aug])
            x_laugs['orig'] = x_hat['orig']
            x_laugs = {aug: model.decoders[decoder](z['orig'] @ model.Laugs[i]) for i,aug in enumerate(augs)}
            if len(model.Laugs) == 2:
                x_laugs['compose'] = model.decoders[decoder](z['orig']@ model.Laugs[0] @ model.Laugs[1])
            
            test_loss += sum([vae_recon_loss(imgs[aug], x_laugs[aug], image_size=args.IMG_DIM) for aug in x_laugs.keys()])

    test_loss /= len(test_loader.dataset)
    return test_loss


def test_augmentation(model, test_loader, device, aug_list = [0,1,2,3,4,5,6], vae=False, image_dim=28, decoder=1):
    model.eval()
    test_loss = {}
    with torch.no_grad():
        for idx, x in enumerate(test_loader):

            for aug in x.keys():x[aug] = x[aug].to(device)

            x_augs = {}

            if vae:
                for aug in x.keys(): x_augs[aug], _, _ = model(x[aug])
            else:
                _, z, _ = model(x['orig'])
                x['r_compose'] = x['compose']
                x['inverse'] = x['']

            if idx == 0:
                for aug in x.keys(): test_loss[aug] = 0
                test_loss['Total'] = 0
                aug_names = [*x.keys()]

            if not vae: 
                if 0 in aug_list:
                    x_augs[aug_names[0]] = model.decoders[decoder](z)
                if 1 in aug_list:
                    x_augs[aug_names[1]] = model.decoders[decoder](z @ model.Laugs[0])
                if 2 in aug_list:
                    x_augs[aug_names[2]] = model.decoders[decoder](z @ model.Laugs[1])
                if 3 in aug_list: 
                    x_augs[aug_names[3]] = model.decoders[decoder](z @ model.Laugs[0] @ model.Laugs[1])
                if 4 in aug_list:
                    x_augs[aug_names[4]] = model.decoders[decoder](z @ model.Laugs[1] @ model.Laugs[0])

            list_losses = [vae_recon_loss(x[aug], x_augs[aug], image_size=image_dim) for aug in x_augs.keys()]

            for i,aug in enumerate(aug_names):
                test_loss[aug] += list_losses[i].item()

            test_loss['Total'] += sum(list_losses).item()

    for aug in aug_names + ['Total']:
        test_loss[aug] /= len(test_loader.dataset)

    print_string = ''.join(['{}: {:.4f}\n'.format(aug, test_loss[aug]) for aug in aug_names + ['Total']])

    print('Aug Test Losses...\n{}'.format(print_string))

    return test_loss