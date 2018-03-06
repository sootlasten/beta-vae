import functools

import tensorflow as tf
from tensorflow.contrib.layers import fully_connected


class DenseVAE():
    def __init__(self, x, z, nb_latents, beta, eta):
        self.x = x
        self.z = z

        self.nb_latents = nb_latents
        self.beta = beta
        self.eta = eta
    
        # call stuff
        self._reconstruct = None
        self._encode = None
        self._decode = None
        self._optimize = None
        
        self.encode
        self.reconstruct
        self.decode
        self.optimize
        
    @property
    def reconstruct(self):
        if self._reconstruct is None:
            mean, logvar = self.encode
            eps = tf.random_normal(tf.shape(mean))
            z = mean + tf.sqrt(tf.exp(logvar))*eps
            self._reconstruct = self.__decode(z)
        return self._reconstruct
    
    @property
    def encode(self):
        if self._encode is None:
            fc1 = fully_connected(self.x, 1200)
            fc2 = fully_connected(fc1, 1200)
            mean = fully_connected(fc2, self.nb_latents, activation_fn=None)
            logvar = fully_connected(fc2, self.nb_latents, activation_fn=None)
            self._encode = mean, logvar
        return self._encode
    
    @property
    def decode(self):
        if self._decode is None:
            self._decode = self.__decode(self.z, reuse=True)
        return self._decode
    
    def __decode(self, z, reuse=False):
        with tf.variable_scope("decoder", reuse=reuse):
            fc1 = fully_connected(z, 1200, activation_fn=tf.nn.tanh)
            fc2 = fully_connected(fc1, 1200, activation_fn=tf.nn.tanh)
            fc3 = fully_connected(fc2, 1200, activation_fn=tf.nn.tanh)
            recon = fully_connected(fc3, 4096, activation_fn=None)
        return recon
    
    @property
    def optimize(self):
        if self._optimize is None:
            ce = tf.reduce_mean(tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(
                    labels=self.x, logits=self.reconstruct), axis=1))
            mean, logvar = self.encode
            kl = -self.beta*0.5*(1 + logvar - tf.exp(logvar) - tf.square(mean))
            loss = ce + tf.reduce_mean(tf.reduce_sum(kl, axis=1))
            optimizer = tf.train.AdagradOptimizer(self.eta)
            self._optimize = optimizer.minimize(loss), loss, tf.reduce_mean(kl, axis=0)
        return self._optimize

