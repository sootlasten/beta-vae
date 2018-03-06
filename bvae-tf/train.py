import os
import shutil
import time
import argparse

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from models import DenseVAE
from data_utils import SpritesDataset


RESULTS_DIR = 'results'

def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def _make_results_dir(dirpath=RESULTS_DIR):
    if os.path.isdir(dirpath):
        shutil.rmtree(dirpath)
    os.makedirs(dirpath)


def train(args):
    _make_results_dir()
        
    sprites = SpritesDataset(args.batch_size)
    testpoint = sprites.imgs[0].flatten()[np.newaxis, :]

    x = tf.placeholder(tf.float32, [None, 4096])
    z = tf.placeholder(tf.float32, [None, args.nb_latents])
    model = DenseVAE(x, z, args.nb_latents, args.beta, args.eta)

    runloss, runkl = None, np.array([])
    start_time = time.time()
        
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        
        for epoch_nb in range(1, args.epochs + 1):
            datagen = sprites.gen()

            for batch_idx, batch_xs in enumerate(datagen):
                batch_xs = batch_xs.reshape(args.batch_size, -1)
                _, loss, kl = sess.run(model.optimize, feed_dict={x: batch_xs})

                runloss = loss if not runloss else runloss*0.99 + loss*0.01
                runkl = np.zeros(args.nb_latents) if not len(runkl) else runkl*0.99 + kl*0.01
            
                if not batch_idx % args.log_interval:
                    print("Epoch {}, batch: {}/{} ({:.2f} s), loss: {:.2f}, kl: [{}]".format(
                        epoch_nb, batch_idx, sprites.nb_batches, time.time() - start_time, runloss, 
                        ", ".join("{:.2f}".format(kl) for kl in runkl)))
                    start_time = time.time()
        
                # latent traversal
                if not batch_idx % args.save_interval:
                    mu, _ = sess.run(model.encode, feed_dict={x: testpoint})
                    mus = np.tile(mu, (args.nb_trav, 1))

                    fig = plt.figure(figsize=(10,10))
                    for zi in range(args.nb_latents):
                        muc = np.matrix.copy(mus)
                        muc[:, zi] = np.linspace(-3, 3, args.nb_trav)
                        recon = sess.run(model.decode, feed_dict={z: muc})
                        recon = sigmoid(recon)
                
                        for i in range(args.nb_trav):
                            ax = plt.subplot(args.nb_trav, args.nb_latents, i*args.nb_latents + (zi+1))
                            plt.axis('off')
                            ax.imshow(recon[i].reshape(64, 64), cmap='gray')
                            
                    filename = os.path.join(RESULTS_DIR, 'traversal_' + str(epoch_nb) + '_' + str(batch_idx) + '.png')
                    plt.savefig(filename)


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
    parser.add_argument('--log-interval', type=int, default=100, metavar='N', 
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-interval', type=int, default=1000, metavar='T',
                        help="how many batches to wait before saving latent traversal")
    parser.add_argument('--nb_trav', type=int, default=7, metavar='T',
                        help='how many point to choose linearly from interval [-3, 3] for \
                        latent traversal analysis')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse()
    train(args)

