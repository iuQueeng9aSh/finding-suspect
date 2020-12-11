import tensorflow as tf
import os
import pathlib as pl
import numpy as np

@tf.function
def parse_image(path, img_size, lbl_dim):
    lbl = tf.cast( int( tf.strings.split(path, os.sep)[-2] ) - 1, dtype=tf.float32 )
    lbl = tf.reshape( tf.one_hot( [lbl], lbl_dim ), (lbl_dim,) )
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img,channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)
    img = img[45:-45, 25:-25, :]
    img = tf.image.resize(img, img_size) 
    img = tf.reshape( img, (np.product(img_size)*3,) )
    return tf.concat( [img, lbl], axis=0)

def get_lbl_dim(path):
    return len([ s.name for s in pl.Path(path).glob('*') ])

def create_img_ds(path, img_size):
    lbl_dim = get_lbl_dim(path)
    ds = tf.data.Dataset.list_files( path + '/*/*' )
    return ds.map( lambda p: parse_image(p, img_size, lbl_dim) )
