import os
import shutil
import time
import argparse
import numpy as np
import matplotlib.pyplot as plt 

import torch
from torch import optim
from torch.autograd import Variable
from torch.nn import functional as F
from torchvision.utils import save_image

from models import DenseVAE, ConvVAE
from data_utils import get_dataloader


def _make_results_dir(dirpath='results'):
    if os.path.isdir('results'):
        shutil.rmtree('results')
    os.makedirs('results')


def _traverse_latents(model, datapoint, nb_latents, epoch_nb, batch_idx, dirpath='results'):
    model.eval()

    if isinstance(model, ConvVAE):
        datapoint = datapoint.unsqueeze(0).unsqueeze(1)
        mu, _ = model.encode(datapoint)
    else:
        mu, _ = model.encode(datapoint.view(-1))

    recons = torch.zeros((5, nb_latents, 64, 64))
    for zi in range(nb_latents):
       muc = mu.squeeze().clone()
       for i, val in enumerate(np.linspace(-3, 3, 5)):
           muc[zi] = val
           recon = model.decode(muc).cpu()
           recons[i, zi] = recon.view(64, 64)
           
    filename = os.path.join(dirpath, 'traversal_' + str(epoch_nb) + '_' + str(batch_idx) + '.png')
    save_image(recons.view(-1, 1, 64, 64), filename, nrow=nb_latents, pad_value=1)


def _loss_function(recon_x, x, mu, logvar, beta):
    """Reconstruction + KL divergence losses summed over all elements and batch."""
    bce = F.binary_cross_entropy(recon_x, x.view(-1, 64*64), size_average=False)
    kld = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
    return kld.mean(dim=0), bce + beta*kld.sum()


def train(args):
    _make_results_dir()
    dataloader = get_dataloader(args.batch_size)
    testpoint = torch.Tensor(dataloader.dataset[0])
    if not args.no_cuda: testpoint = testpoint.cuda()

    model = DenseVAE(args.nb_latents)
    model.train()
    if not args.no_cuda: model.cuda()
    optimizer = optim.Adagrad(model.parameters(), lr=args.eta)
    
    runloss, runkld = None, np.array([])
    start_time = time.time()
    
    for epoch_nb in range(1, args.epochs + 1):

        for batch_idx, data in enumerate(dataloader):
            if not args.no_cuda: data = data.cuda()
            
            recon_batch, mu, logvar = model(data)
            kld, loss = _loss_function(recon_batch, data, mu, logvar, args.beta)

            # param update
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss /= len(data)
            runloss = loss if not runloss else runloss*0.99 + loss*0.01
            runkld = np.zeros(args.nb_latents) if not len(runkld) else runkld*0.99 + kld.data.cpu().numpy()*0.01
    
            if not batch_idx % args.log_interval:
                print("Epoch {}, batch: {}/{} ({:.2f} s), loss: {:.2f}, kl: [{}]".format(
                    epoch_nb, batch_idx, len(dataloader), time.time() - start_time, runloss, 
                    ", ".join("{:.2f}".format(kl) for kl in runkld)))
                start_time = time.time()
        
            if not batch_idx % args.save_interval:
                _traverse_latents(model, testpoint, args.nb_latents, epoch_nb, batch_idx)
                model.train()
        

def parse():
    parser = argparse.ArgumentParser(description='train beta-VAE on the sprites dataset')
    parser.add_argument('--eta', type=float, default=1e-2, metavar='L',
                        help='learning rate for Adam (default: 1e-2)')
    parser.add_argument('--beta', type=int, default=4, metavar='B',
                        help='the beta coefficient (default: 4)')
    parser.add_argument('--nb-latents', type=int, default=10, metavar='N',
                        help='number of latents (default: 10)')
    parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                        help='input batch size for training (default: 128)')
    parser.add_argument('--epochs', type=int, default=100, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='enables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=100, metavar='N', 
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-interval', type=int, default=1000, metavar='T',
                        help="how many batches to wait before saving latent traversal")
    return parser.parse_args()


if __name__ == '__main__':
    args = parse()
    train(args)

