#from __future__ import annotations
import tensorflow as tf

from tensorflow.data import Dataset

from pathlib import Path
from typing import Callable, Optional, Union, TYPE_CHECKING
from common import load_patch
import numpy as np
import pandas as pd
from environmental_raster import PatchExtractor
import tifffile
# The following functions can be used to convert a value to a type compatible
# with tf.train.Example.

def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy() # BytesList won't unpack a string from an EagerTensor.
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _float_feature(value):
    """Returns a float_list from a float / double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

def _int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))



def image_example(observation_id, coordinates, label, root="/network/scratch/s/sara.ebrahim-elkafrawy/"):
    observation_id = str(observation_id)

    region_id = observation_id[0]
    if region_id == "1":
        region = "patches-fr"
    elif region_id == "2":
        region = "patches-us"
    else:
        raise ValueError(
            "Incorrect 'observation_id' {}, can not extract region id from it".format(
                observation_id
            )
        )
    subfolder1 = observation_id[-2:]
    subfolder2 = observation_id[-4:-2]

    filename = Path(root) / region / subfolder1 / subfolder2 / observation_id



    rgb_filename = filename.with_name(filename.stem + "_rgb.jpg")
    rgb = open(rgb_filename, 'rb').read()
    image_shape = tf.io.decode_jpeg(rgb).shape
    
    nir_filename = filename.with_name(filename.stem + "_near_ir.jpg")
    near_ir = open(nir_filename, 'rb').read()
    
    altitude_filename = filename.with_name(filename.stem + "_altitude.tif")
    altitude= tifffile.imread(altitude_filename).tobytes()
    
    landcover_filename = filename.with_name(filename.stem + "_landcover.tif")
    landcover=  tifffile.imread(landcover_filename).tobytes()  #open(landcover_filename, 'rb').read()


    feature = {
        'observation_id': _bytes_feature(bytes(observation_id, 'utf-8')),
        'latitude': _float_feature(float(coordinates[0])),
        'longitude': _float_feature(float(coordinates[1])),
        'target': _int64_feature(int(label)),
        'rgb': _bytes_feature(rgb),
        'height': _int64_feature(image_shape[0]),
        'width': _int64_feature(image_shape[1]),
        
        'near_ir': _bytes_feature(near_ir),
        'altitude': _bytes_feature(altitude),
        'landcover': _bytes_feature(landcover),
    }
    
    return tf.train.Example(features=tf.train.Features(feature=feature))

def test_image_example(observation_id, coordinates,  root="/network/scratch/s/sara.ebrahim-elkafrawy/"):
    observation_id = str(observation_id)

    region_id = observation_id[0]
    if region_id == "1":
        region = "patches-fr"
    elif region_id == "2":
        region = "patches-us"
    else:
        raise ValueError(
            "Incorrect 'observation_id' {}, can not extract region id from it".format(
                observation_id
            )
        )
    subfolder1 = observation_id[-2:]
    subfolder2 = observation_id[-4:-2]

    filename = Path(root) / region / subfolder1 / subfolder2 / observation_id



    rgb_filename = filename.with_name(filename.stem + "_rgb.jpg")
    rgb = open(rgb_filename, 'rb').read()
    image_shape = tf.io.decode_jpeg(rgb).shape
    
    nir_filename = filename.with_name(filename.stem + "_near_ir.jpg")
    near_ir = open(nir_filename, 'rb').read()
    
    altitude_filename = filename.with_name(filename.stem + "_altitude.tif")
    altitude= tifffile.imread(altitude_filename).tobytes() #open(altitude_filename, 'rb').read()
    
    landcover_filename = filename.with_name(filename.stem + "_landcover.tif")
    landcover= tifffile.imread(landcover_filename).tobytes() #open(landcover_filename, 'rb').read()


    feature = {
        'observation_id': _bytes_feature(bytes(observation_id, 'utf-8')),
        'latitude': _float_feature(coordinates[0]),
        'longitude': _float_feature(coordinates[1]),
        'rgb': _bytes_feature(rgb),
        'height': _int64_feature(image_shape[0]),
        'width': _int64_feature(image_shape[1]),
        
        'near_ir': _bytes_feature(near_ir),
        'altitude': _bytes_feature(altitude),
        'landcover': _bytes_feature(landcover),
    }
    return tf.train.Example(features=tf.train.Features(feature=feature))

@tf.function
def init_dataset(root: Union[str, Path],
        subset: str,
        *,
        region: str = "both",
        patch_data: str = "all",
        use_rasters: bool = True,
        patch_extractor: Optional[PatchExtractor] = None,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,):
     
    possible_subsets = ["train", "val", "train+val", "test"]
    if subset not in possible_subsets:
        raise ValueError(
                "Possible values for 'subset' are: {} (given {})".format(
                    possible_subsets, subset
                )
            )

    possible_regions = ["both", "fr", "us"]
    if region not in possible_regions:
            raise ValueError(
                "Possible values for 'region' are: {} (given {})".format(
                    possible_regions, region
                )
            )

    if subset == "test":
        subset_file_suffix = "test"
        training_data = False
    else:
        subset_file_suffix = "train"
        training_data = True

    df_fr = pd.read_csv(
        root
        / "observations"
        / "observations_fr_{}.csv".format(subset_file_suffix),
        sep=";",
        index_col="observation_id",
        )
    df_us = pd.read_csv(
        root
        / "observations"
        / "observations_us_{}.csv".format(subset_file_suffix),
        sep=";",
        index_col="observation_id",
        )

    if region == "both":
        df = pd.concat((df_fr, df_us))
    elif region == "fr":
        df = df_fr
    elif region == "us":
        df = df_us

    if training_data and subset != "train+val":
        ind = df.index[df["subset"] == subset]
        df = df.loc[ind]

    observation_ids = df.index
    coordinates = df[["latitude", "longitude"]].values

    if training_data:
        targets = df["species_id"].values
    else:
        targets = None

        # FIXME: add back landcover one hot encoding?
        # self.one_hot_size = 34
        # self.one_hot = np.eye(self.one_hot_size)


    return(observation_ids, coordinates, targets)

if __name__=="__main__":

    init = init_dataset(Path("/network/scratch/s/sara.ebrahim-elkafrawy/", region="both"),
            subset="train")

    observation_ids, coordinates, targets= init
    record_file = '/network/scratch/t/tengmeli/temp_glc/train_images_new6.tfrecords'
    with tf.io.TFRecordWriter(record_file) as writer:
        for i in range (len(observation_ids)):
            
            #tf_example = test_image_example(str(observation_ids[i].numpy()), coordinates[i], root="/network/scratch/s/sara.ebrahim-elkafrawy/")
            tf_example = image_example(str(observation_ids[i].numpy()), coordinates[i], targets[i], root="/network/scratch/s/sara.ebrahim-elkafrawy/")
            writer.write(tf_example.SerializeToString())
    print("done")
