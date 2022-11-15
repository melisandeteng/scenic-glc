from __future__ import annotations
from pathlib import Path
from typing import Callable, Optional, Union, TYPE_CHECKING

import numpy as np
import pandas as pd
import tensorflow_io as tfio

from .common import load_patch, get_num_channels

import tensorflow_datasets as tfds
import tensorflow as tf



def parse_tfr_element(element, bands="all", subset="train"):
    #use the same structure as above; it's kinda an outline of the structure we now want to create
    # Create a dictionary describing the features.
    data = {
        'observation_id': tf.io.FixedLenFeature([], tf.string),
        'latitude': tf.io.FixedLenFeature([], tf.float32),
        'longitude': tf.io.FixedLenFeature([], tf.float32),
        'rgb': tf.io.FixedLenFeature([], tf.string),
        'height': tf.io.FixedLenFeature([], tf.int64),
        'width': tf.io.FixedLenFeature([], tf.int64),
        'near_ir': tf.io.FixedLenFeature([], tf.string),
        'altitude': tf.io.FixedLenFeature([], tf.string),
        'landcover': tf.io.FixedLenFeature([], tf.string),
    }
    
    if subset!="test":
        data['target'] = tf.io.FixedLenFeature([], tf.int64)
        
    content = tf.io.parse_single_example(element, data)

    height = content['height']
    width = content['width']
    raw_image = content['rgb']
    if subset!="test":
        #label = tf.expand_dims(content['target'], axis=0)
        label =content['target']
        
    #get our 'feature'-- our image -- and reshape it appropriately
    feature_rgb = tf.image.decode_image(raw_image, channels=3)
    feature_nir= tf.image.decode_image(content["near_ir"], channels=1)
    feature_landcover=  tf.expand_dims(tf.reshape(tf.io.decode_raw(content["landcover"], tf.uint8), shape=[256,256]), -1)#tf.expand_dims(tfio.experimental.image.decode_tiff(content["landcover"])[:,:,0], -1)
    feature_alt = tf.expand_dims(tf.reshape(tf.io.decode_raw(content["altitude"], tf.uint16), shape=[256,256]), -1)
        
    features = {"rgb": (tf.cast(feature_rgb, tf.float32)-[106.9413, 114.8733, 104.5285])/ [51.0005, 44.8595, 43.2014],
                "near_ir": (tf.cast(feature_nir, tf.float32)- 131.0458)/53.0884,
                "landcover": tf.cast(feature_landcover, tf.float32)/33.0, 
                "altitude":(tf.cast(feature_alt, tf.float32)-298.1693)/459.3285}
    if bands==["all"]:
        bands = ["rgb", "near_ir", "landcover", "altitude"]
    features = [features[b] for b in bands]
    features = tf.concat(features, axis = -1)
    #features = tf.expand_dims(features, axis=0)
    if subset=="test":
        return {"inputs":features}
    else:
        return {"inputs": features, "label":label}

def get_dataset(path = '/network/scratch/t/tengmeli/temp_glc/val_images.tfrecords', bands=["all"], subset="train"):
  
    dataset = tf.data.TFRecordDataset([path])

    #pass every single feature through our mapping function
    dataset = dataset.map(
        (lambda x: parse_tfr_element(x, bands, subset))
    )
    
    return dataset

