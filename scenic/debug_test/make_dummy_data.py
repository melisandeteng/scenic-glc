# from __future__ import annotations
import tensorflow as tf

from tensorflow.data import Dataset

from pathlib import Path
from typing import Callable, Optional, Union, TYPE_CHECKING
from scenic.projects.glc.data.common import load_patch
import numpy as np
import pandas as pd
from scenic.projects.glc.data.environmental_raster import PatchExtractor
import tifffile

# The following functions can be used to convert a value to a type compatible
# with tf.train.Example.


def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy()  # BytesList won't unpack a string from an EagerTensor.
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _float_feature(value):
    """Returns a float_list from a float / double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def _int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def image_example(
    observation_id, label=1, root="/network/scratch/s/sara.ebrahim-elkafrawy/",
):
    observation_id = str(observation_id)
    rgb = open("/home/mila/t/tengmeli/GLC/debug_scenic/data/n01580077_jay.jpeg", "rb").read()
               #"/home/mila/t/tengmeli/scenic-glc/scenic/debug_test/data/dummy_rgb.jpg", "rb").read()
    image_shape = tf.io.decode_jpeg(rgb).shape

    nir_filename = "/home/mila/t/tengmeli/scenic-glc/scenic/debug_test/data/dummy_nir.jpg"
    near_ir = open(nir_filename, "rb").read()

    altitude_filename = "/home/mila/t/tengmeli/scenic-glc/scenic/debug_test/data/test.tif"
    altitude = tifffile.imread(altitude_filename).tobytes()

    landcover_filename = "/home/mila/t/tengmeli/scenic-glc/scenic/debug_test/data/test.tif"
    landcover = tifffile.imread(
        landcover_filename
    ).tobytes()  # open(landcover_filename, 'rb').read()
    
    latitude = 45
    longitude = 45
    
    feature = {
        "observation_id": _bytes_feature(bytes(observation_id, "utf-8")),
        "target": _int64_feature(int(label)),
        'latitude': _float_feature(latitude),
        'longitude': _float_feature(longitude),
        "rgb": _bytes_feature(rgb),
        "height": _int64_feature(image_shape[0]),
        "width": _int64_feature(image_shape[1]),
        "near_ir": _bytes_feature(near_ir),
        "altitude": _bytes_feature(altitude),
        "landcover": _bytes_feature(landcover),
    }

    return tf.train.Example(features=tf.train.Features(feature=feature))


def test_image_example(
    observation_id, root="/network/scratch/s/sara.ebrahim-elkafrawy/"
):
    latitude = 45
    longitude = 45
    
    observation_id = str(observation_id)
    rgb = open("/home/mila/t/tengmeli/GGLC/debug_scenic/data/n01580077_jay_1.jpeg", "rb").read() #("/home/mila/t/tengmeli/scenic-glc/scenic/debug_test/data/dummy_rgb.jpg", "rb").read()
    image_shape = tf.io.decode_jpeg(rgb).shape
    
    nir_filename = "/home/mila/t/tengmeli/scenic-glc/scenic/debug_test/data/dummy_nir.jpg"
    near_ir = open(nir_filename, "rb").read()

    altitude_filename = "/home/mila/t/tengmeli/scenic-glc/scenic/debug_test/data/test.tif"
    altitude = tifffile.imread(altitude_filename).tobytes()

    landcover_filename = "/home/mila/t/tengmeli/scenic-glc/scenic/debug_test/data/test.tif"
    landcover = tifffile.imread(
        landcover_filename
    ).tobytes()  # open(landcover_filename, 'rb').read()
    feature = {
        "observation_id": _bytes_feature(bytes(observation_id, "utf-8")),
        "rgb": _bytes_feature(rgb),
        'latitude': _float_feature(latitude),
        'longitude': _float_feature(longitude),
        "height": _int64_feature(image_shape[0]),
        "width": _int64_feature(image_shape[1]),
        "near_ir": _bytes_feature(near_ir),
        "altitude": _bytes_feature(altitude),
        "landcover": _bytes_feature(landcover),
    }
    return tf.train.Example(features=tf.train.Features(feature=feature))


if __name__ == "__main__":

    record_file = "dummy_jay_tfrecords.tfrecords"
    with tf.io.TFRecordWriter(record_file) as writer:
        for i in range(16):
            observation_id = str(i)
            # tf_example = test_image_example(str(observation_ids[i].numpy()), coordinates[i], root="/network/scratch/s/sara.ebrahim-elkafrawy/")
            tf_example = image_example(observation_id,)
            writer.write(tf_example.SerializeToString())
    print("done")
