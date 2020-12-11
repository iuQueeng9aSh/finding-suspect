import tensorflow as tf
import keras
from keras.layers import Dense, Lambda, Activation, Reshape, Dropout, Flatten, concatenate
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D, Deconv2D
from keras.models import Sequential
from keras import backend as K
import numpy as np
import pickle

class VAE_Encoder(tf.keras.Model):

    def __init__(self, image_size, filters, latent=8, dropout=0.2):
        super(VAE_Encoder, self).__init__()

        self.dropout = Dropout(dropout)
        self.img = Sequential( [Reshape( (image_size[0], image_size[1], 3) )] )
        for f in filters:
            self.img.add(Conv2D(filters=f, kernel_size=5, strides=2, padding='same', use_bias=False ) )
            self.img.add(BatchNormalization(axis=3))
            self.img.add(Activation(K.elu))
        self.img.add(Flatten())
        self.enc = Sequential(
            [
                Dense(1024),
                BatchNormalization(),
                Activation(K.elu),
                Dense(latent*2),
            ]
        )

    def call(self, inputs, training=None):
        X, lbl = inputs
        if training:
            X = self.dropout(X)
        t = concatenate( [self.img( X ), lbl ], axis=1 )
        return self.enc(t)        

class VAE_Decoder(tf.keras.Model):

    def __init__(self, image_size, filters, lbl_dim=1, latent=8, flows=0, dropout=0.2, use_beta=False):
        super(VAE_Decoder, self).__init__()
        
        w = image_size[0]//2**len(filters)
        h = image_size[1]//2**len(filters)

        fclSiz = w * h * filters[0]
        self.image_size = image_size
        self.lbl_dim = lbl_dim
        self.latent = latent
        self.flows = flows
        self.use_beta = use_beta
        self.flow = Sequential()
        for f in range(flows):
            self.flow.add( Flow(latent) )
        self.dec = Sequential(
            [
                Dense(fclSiz),
                BatchNormalization(),
                Activation(K.elu),
                Reshape( [w, h, filters[0]] ),
            ]
        )
        for f in filters:
            self.dec.add(Deconv2D(filters=f, kernel_size=5, strides=2, padding='same', use_bias=False))
            self.dec.add(BatchNormalization(axis=3))
            self.dec.add(Activation(K.elu))
        self.dropout = Dropout(dropout)
        self.img = Sequential(
            [
                # Deconv2D(filters=6, kernel_size=5, strides=1, padding='same', activation=K.tanh ),
                Deconv2D(filters=6, kernel_size=5, strides=1, padding='same' ),
                Reshape([image_size[0] * image_size[1] * 6])
            ]
        )
            
    def call(self, inputs, training=None):
        t, lbl = inputs
        t = self.flow(t)
        Xd = self.dec(concatenate([t, lbl], axis=1))
        if training:
            Xd = self.dropout(Xd)
        return self.img(Xd)
    
    def sample(self, number_samples=1, lbl=None, flow=True, sdvs=None):
        t = tf.reshape(
            tf.random.normal( (self.latent * number_samples,) ),
            (number_samples, self.latent) 
        )
        if not sdvs is None:
            assert len(sdvs) == self.latent, "invalid vector of standard deviations"
            sdvs = tf.reshape( tf.convert_to_tensor(sdvs), [1, self.latent] )
            sdvs = tf.repeat(sdvs, number_samples, axis=0)
            sdvs = tf.cast( sdvs, tf.float32 )
            t = sdvs * t
        if flow:
            t = self.flow(t)
        lbl_dim = self.lbl_dim
        if lbl_dim > 1:
            if lbl is None:
                idxs = np.random.randint(0, high=lbl_dim, size=number_samples)
                lbl = tf.one_hot([idxs], lbl_dim)
                lbl = tf.reshape( lbl, (number_samples, lbl_dim) )
            elif isinstance(lbl, int):
                assert lbl >= 0 and lbl < lbl_dim, "invalid label: " + str(lbl)
                lbl = tf.one_hot( [lbl], lbl_dim )
                lbl = tf.repeat( lbl, number_samples, axis=0 )
            else:
                assert len(lbl) == lbl_dim, "generate_images - invalid label size"
                assert np.abs(np.sum(lbl) - 1.0) < 1.0e-8, "generate_images - label vector must add up to one"
                lbl = tf.reshape( tf.convert_to_tensor(lbl), (1, lbl_dim) )
                lbl = tf.repeat( lbl, number_samples, axis=0 )
        else:
            lbl = tf.ones( [number_samples, 1] )
        return tf.concat( [t, lbl], axis=1 )
    
    def image_from_sample( self, sample, flow=True ):
        image_dim = np.product( self.image_size )
        if not flow:
            t = self.flow( sample[:,:self.latent] )
            lbl = sample[:,self.latent:]
            sample = tf.concat( [t, lbl], axis=1 )
        if self.use_beta:
            Xd = tf.exp( self.img(self.dec(sample)) )
            alpha = Xd[:, :image_dim]
            beta = Xd[:, image_dim:]
            Xd = alpha / (alpha + beta)
        else:
            Xd = self.img(self.dec(sample))[:,:image_dim]
        return tf.reshape( Xd, [len(sample)] + self.image_size ).numpy()

    def generate_images( self, number_images, lbl=None, flow=True, with_samples=False, sdvs=None ):
        t = self.sample( number_images, lbl=lbl, flow=flow, sdvs=sdvs )
        imgs = self.image_from_sample(t, flow=flow)
        if with_samples:
            return t, imgs
        return imgs

@tf.function
def sample(mean_log_var):
    mean, log_var = mean_log_var
    epsilon = K.random_normal( shape=tf.shape(mean) )
    z = epsilon * K.exp(0.5*log_var) + mean
    return z

@tf.function
def neg_vlb_gaussian(inputs, Xd, img_dim, t_mean, t_log_var, itrCnt):
    X = inputs[:, :img_dim]
    Xd_mean = Xd[:, :img_dim]
    Xd_log_var = Xd[:, img_dim:]
    negGLL = tf.math.reduce_mean(
        tf.math.reduce_sum(
            Xd_log_var + tf.square( X - Xd_mean ) * tf.exp( -Xd_log_var ),
            1
        )
    )    
    KLD = tf.math.reduce_mean(
        tf.math.reduce_sum(
            tf.square( t_mean ) + tf.exp( t_log_var ) - t_log_var - 1.0,
            1
        )
    )
    p = tf.minimum( 1.0, 0.01 + tf.cast( itrCnt, dtype=tf.float32 ) / 10000.0 )
    return p * negGLL + KLD

def vae_loss_gaussian(mdl):
    
    def loss( X, Xd ):
        return neg_vlb_gaussian(X, Xd, mdl.img_dim, mdl.t_mean, mdl.t_log_var, mdl.optimizer.iterations)
    
    return loss

@tf.function
def neg_vlb_beta(inputs, Xd, img_dim, t_mean, t_log_var, itrCnt):
    X = inputs[:,:img_dim]
    Xd_alpha = tf.maximum( 1.0e-4, tf.exp( Xd[:, :img_dim] ) )
    Xd_beta  = tf.maximum( 1.0e-4, tf.exp( Xd[:, img_dim:] ) )
    logX1 = tf.math.log( tf.maximum( 1.0e-4, X ) )
    logX2 = tf.math.log( tf.maximum( 1.0e-4, 1.0 - X ) )
    negBLL = tf.math.reduce_mean(
        tf.math.reduce_sum(
            tf.math.lgamma( Xd_alpha ) + tf.math.lgamma( Xd_beta ) - 
            tf.math.lgamma( Xd_alpha + Xd_beta ) -
            ( Xd_alpha - 1.0 ) * logX1 - ( Xd_beta - 1.0 ) * logX2,
            1
        )
    )    
    KLD = 0.5 * tf.math.reduce_mean(
        tf.math.reduce_sum(
            tf.square( t_mean ) + tf.exp( t_log_var ) - t_log_var - 1.0,
            1
        )
    )
    p = tf.minimum( 1.0, 0.01 + tf.cast( itrCnt, dtype=tf.float32 ) / 10000.0 )
    return p * negBLL + KLD

def vae_loss_beta(mdl):
    
    def loss( X, Xd ):
        return neg_vlb_beta(X, Xd, mdl.img_dim, mdl.t_mean, mdl.t_log_var, mdl.optimizer.iterations)
    
    return loss

class NormFlowInitializer(tf.keras.initializers.Initializer):

    def __init__(self, w):
        self.w = w
    
    def __call__(self, shape, dtype=None):
        u = tf.random.uniform( shape, minval=-1.0, maxval=1.0, dtype=dtype)
        dot = tf.tensordot( self.w, u, axes=1 )
        if ( dot >= -1.0 ):
            return u
        nrm = ( tf.math.softplus( dot ) - dot - 1. ) / tf.norm( self.w )
        return u + nrm * self.w

class Flow(tf.keras.layers.Layer):
    
    def __init__(self, dim, **kwargs ):
        super( Flow, self ).__init__(**kwargs)
        self.dim = dim
        
    def build(self, input_shape):
        self.w = self.add_weight(
            shape=(self.dim,),
            initializer=tf.keras.initializers.RandomUniform(minval=-1.0, maxval=1.0),
            trainable=True,
        )
        self.u = self.add_weight(
            shape=(self.dim,),
            initializer=NormFlowInitializer(self.w),
            trainable=True,
        )
        self.b = self.add_weight(
            shape=(1,),
            initializer=tf.keras.initializers.RandomUniform(minval=-1.0, maxval=1.0),
            trainable=True,
        )
        tf.debugging.Assert(tf.greater_equal( tf.tensordot( self.w, self.u, axes=1 ), -1.), [self.w, self.u])

    def call(self, inputs):
        dot = tf.tensordot( self.w, self.u, axes=1 )
        tf.debugging.Assert(tf.greater_equal( dot, -1.), [self.w, self.u])
        psi = tf.tensordot(inputs, self.w, axes=1) + self.b
        self.add_loss(
            -1.0 * tf.reduce_mean(
                tf.math.log( 
                    tf.abs( 1. + dot * ( 1.0 - tf.square( tf.math.tanh(psi) ) ) )
                )
            )
        )
        return inputs + tf.tensordot( tf.math.tanh( psi ), self.u, axes=0 )

    def get_config(self):
        config = super(Flow, self).get_config()
        config.update({"dim": self.dim})
        return config

class VAE(tf.keras.Model):
    
    def __init__(self, image_size, filters, lbl_dim=1, latent=8, flows=0, dropout=0.2, use_beta=False):
        super(VAE, self).__init__()
        if isinstance(image_size, int):
            image_size = (image_size, image_size)
        self.filters = filters.copy()
        self.dropout = dropout
        self.img_dim = image_size[0] * image_size[1] * 3
        self.get_X = Lambda(lambda x: x[:, :self.img_dim] )
        self.get_lbl = Lambda(lambda x: x[:, self.img_dim:] )
        self.encoder = VAE_Encoder([image_size[0], image_size[1], 3],
                        filters,
                        latent=latent,
                        dropout=dropout)
        filters.reverse()
        self.decoder = VAE_Decoder([image_size[0], image_size[1], 3],
                        filters,
                        lbl_dim=lbl_dim,
                        latent=latent,
                        flows=flows,
                        dropout=dropout,
                        use_beta=use_beta)
        self.get_t_mean = Lambda(lambda h: h[:, :latent])
        self.get_t_log_var = Lambda(lambda h: h[:, latent:])
        self.sample_t = Lambda(sample)
            
    def call(self, inputs, training=None):
        #encode
        X = self.get_X(inputs)
        lbl = self.get_lbl(inputs)
        h = self.encoder( [X, lbl], training=training)
        # sample
        self.t_mean = self.get_t_mean(h)
        self.t_log_var = self.get_t_log_var(h)
        t = self.sample_t([self.t_mean, self.t_log_var])
        # decode
        return self.decoder( [t, lbl], training=training )

    def save(self, path, with_opt=False):
        dat = ( self.decoder.image_size,
                self.filters,
                self.decoder.lbl_dim,
                self.decoder.latent,
                self.decoder.flows,
                self.dropout,
                self.decoder.use_beta,
                tf.keras.backend.get_value( self.optimizer.lr ),
                self.get_weights() )
        if with_opt:
            dat = dat + (self.optimizer.get_weights(),)
        with open(path, 'wb') as f:
            pickle.dump(dat, f)
            f.close()
    
    def latent(self, inputs, flow=True, sample=False):
        X = self.get_X(inputs)
        lbl = self.get_lbl(inputs)
        h = self.encoder( [X, lbl] )
        t = self.get_t_mean( h )
        if sample:
            log_var = self.get_t_log_var(h)            
            epsilon = K.random_normal( shape=tf.shape(t) )
            t = epsilon * K.exp(0.5*log_var) + t
        if flow:
            t = self.decoder.flow( t )
        return tf.concat( [t, lbl], axis=1 )

    def KLD(self, image, label ):
        X = tf.cast( tf.reshape( image, [1, self.img_dim] ), tf.float32 )
        lbl = tf.cast( tf.reshape( label, [1, self.decoder.lbl_dim] ), tf.float32 )
        h = self.encoder( [X, lbl] )
        t_mean = self.get_t_mean(h)
        t_log_var = self.get_t_log_var(h)
        KLD = 0.5 * tf.math.reduce_sum(
            tf.square( t_mean ) + tf.exp( t_log_var ) - t_log_var - 1.0,
            1
        )
        return KLD.numpy()[0]
    
def create_vae(
        image_size, filters, lbl_dim=1, latent=8, flows=0, dropout=0.2, learning_rate=0.0001,
        mdl_weights=None, opt_weights=None, use_beta=False
    ):
    """
    Constructs VAE model with given parameters.
    :param image_size: size of input image
    :param filters: list of filter nambers for each convolutional layer of encoder and decoder (reversed)
    :param lbl_dim: dimensionality of the label space (one-hot, 1 for no labels)
    :param latent: latent space dimension
    :param flows: number of finite normalizing flow layers
    :param dropout: dropout rate for input layer of both encoder and decoder
    :param learning_rate: the learning rate for the ADAM optimizer
    :param mdl_weights: the weights of a pre-trained model
    :param opt_weights: the weights of an optimizer to continue training
    :param use_beta: if True, model the posterior as factorized Beta distribution (instead of Gaussian)
    Returns compiled Keras model along with encoder and decoder
    """
    if isinstance(image_size, int):
        image_size = (image_size, image_size)
    mdl = VAE( image_size, filters, lbl_dim=lbl_dim, latent=latent, flows=flows, dropout=dropout, use_beta=use_beta )
    mdl( tf.ones( shape=(1, image_size[0]*image_size[1]*3 + lbl_dim ) ) )

    opt = keras.optimizers.Adam(lr=learning_rate)

    if opt_weights:
        grad_vars = mdl.trainable_weights
        zero_grads = [tf.zeros_like(w) for w in grad_vars]
        opt.apply_gradients(zip(zero_grads, grad_vars))
        opt.set_weights(opt_weights)
    
    if mdl_weights:
        mdl.set_weights( mdl_weights )

    if use_beta:
        mdl.compile( optimizer=opt, loss=vae_loss_beta(mdl) )
    else: 
        mdl.compile( optimizer=opt, loss=vae_loss_gaussian(mdl) )
    return mdl

def load_vae( path ):
    try:
        with open(path, 'rb') as f:
            dat = pickle.load(f)
            f.close()
            image_size, filters, lbl_dim, latent, flows, dropout, use_beta, \
                learning_rate, weights = dat[:10]
            if len(dat) == 10:
                opt_weights = dat[-1]
            else:
                opt_weights = None
    except:
        print( "Failed to load model data from " + path )
        return
    mdl = create_vae(
        image_size,
        filters,
        lbl_dim=lbl_dim,
        latent=latent,
        flows=flows,
        dropout=dropout,
        learning_rate=learning_rate,
        mdl_weights=weights,
        opt_weights=opt_weights,
        use_beta=use_beta )
    return mdl
