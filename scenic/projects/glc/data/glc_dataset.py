"""Data-loader to read from SSTables using the MediaSequence format."""

import functools
import os
from typing import Dict, Iterator, List, Optional, Text, Tuple, Union

import abc
from absl import logging
from flax import jax_utils
import jax
import jax.numpy as jnp
import ml_collections
import numpy as np
from scenic.dataset_lib import dataset_utils
from scenic.dataset_lib import datasets
from scenic.projects.glc.data import common
import tensorflow as tf
from scenic.projects.glc.data import tf_dataset
from scenic.projects.glc.data.common import get_num_channels
# Aliases for custom types:
Batch = Dict[str, jnp.ndarray]
Rng = Union[jnp.ndarray, Dict[str, jnp.ndarray]]

TRAIN_IMAGES = 1587395
EVAL_IMAGES = 40080
TEST_IMAGES = 36421


class TFRecordDatasetFactory(abc.ABC):
    """Reader for TFRecords

    Attributes:
      num_classes: int. The number of classes in the dataset.
      base_dir: str. The base directory from which the TFRecords are read.
      subset: str. The subset of the dataset. In Scenic, the subsets are
        "train", "validation" and "test".
    """

    def __init__(
            self,
            base_dir: str,
            tables: Dict[str, Union[str, List[str]]],
            examples_per_subset: Dict[str, int],
            num_classes: int,
            subset: str = 'train',
            fraction_data: float = 1.0):
        """Initializes the instance of TFRecordDatasetFactory.

        Initializes a data-loader.

        Args:
          base_dir: The base directory of the TFRecords.
          tables: A dictionary mapping the subset name (train, val or test) to the
            relative path of the TFRecord containing them.
            The values of the dictionary can either be a string or a list. If it is
            a string, it specifies all the shards in the TFRecord.
            Example - "/path/to/tfrecord@10".
            If passing a list, each entry is a shard of the TFRecord.
            Example - "[/path/to/tfrecord_shard_1_of_10, ...,
                        /path/to/tfrecord_shard_10_of_10]."
            The latter scenario is useful for debugging.
          examples_per_subset:  A dictionary mapping the subset name (train, val or
            test) to the number of examples in the dataset for that subset.
          num_classes: The number of classes in the dataset.
          subset: The subset of the dataset to load. Must be a key of "tables"
          fraction_data: The fraction of the data to load. If less than 1.0, this
            fraction of the total TFRecord shards are read.


        Raises:
          ValueError: If subset is not a key of tables or examples_per_subset
        """
        if (subset not in tables) or (subset not in examples_per_subset):
            raise ValueError(f'Invalid subset {subset!r}. '
                             f'The available subsets are: {set(tables)!r}')

        self.num_classes = num_classes
        self.base_dir = base_dir
        self.subset = subset
        self.num_examples = examples_per_subset[subset]

        self.path = os.path.join(self.base_dir, tables[subset])

        super().__init__()


def glc_load_split(dsfactory,
                   batch_size,
                   subset,
                   bands=["rgb", "nir"],
                   dtype=tf.float32,
                   image_size=224,
                   shuffle_seed=None,
                   transform=None):
    """Creates a split from the ImageNet dataset using TensorFlow Datasets.

    For the training set, we drop the last partial batch. This is fine to do
    because we additionally shuffle the data randomly each epoch, thus the trainer
    will see all data in expectation. For the validation set, we pad the final
    batch to the desired batch size.

    Args:
    batch_size: int; The batch size returned by the data pipeline.
    subset: string; Whether to load the train or evaluation split.
    dtype: TF data type; Data type of the image.
    image_size: int; The target size of the images.
    prefetch_buffer_size: int; Buffer size for the TFDS prefetch.
    shuffle_seed: The seed to use when shuffling the train split.
    data_augmentations: fuction #list(str); Types of data augmentation applied on
      training data.

    Returns:
    A `tf.data.Dataset`.
    """
    if subset == "train":
        split_size = TRAIN_IMAGES // jax.process_count()
        start = jax.process_index() * split_size
        split = 'train[{}:{}]'.format(start, start + split_size)
    elif subset == "validation":
        split_size = EVAL_IMAGES // jax.process_count()
        start = jax.process_index() * split_size
        split = 'validation[{}:{}]'.format(start, start + split_size)
    else:
        split_size = TEST_IMAGES // jax.process_count()
        start = jax.process_index() * split_size
        split = 'test[{}:{}]'.format(start, start + split_size)

    a = dsfactory(subset=subset)
    dataset = tf.data.TFRecordDataset([a.path])
    
    # pass every single feature through our mapping function
    dataset = dataset.map(
        (lambda x: tf_dataset.parse_tfr_element(x, bands, subset)))

    

    num_examples = a.num_examples
    options = tf.data.Options()
    options.threading.private_threadpool_size = 48
    dataset = dataset.with_options(options)
    #print("BBBBBBB", next(iter(dataset)))
    #import pdb; pdb.set_trace()
    #dataset = dataset.cache()
    #a = next(iter(dataset))
    if subset == "train":
        dataset = dataset.repeat()
        dataset = dataset.map(lambda x:
                          transform(x))
        dataset = dataset.shuffle(16 * batch_size, seed=shuffle_seed)
        dataset = dataset.batch(batch_size, drop_remainder=subset == "train")
    
    else:
        dataset = dataset.map(lambda x:
                          transform(x))
        dataset = dataset.batch(batch_size, drop_remainder=False)
        dataset = dataset.repeat()
    
    return dataset, num_examples


def get_augmentations(example, onehot_label, num_channels=6, dtype=tf.float32, data_augmentations=None, crop_size=224, subset="train"):
    """Apply data augmentation on the given training example.

    Args:
    example: dict; Example that has an 'image' and a 'label'.
    dtype: Tensorflow data type; Data type of the image.
    data_augmentations: list(str); Types of data augmentation applied on
      training data.

    Returns:
    An augmented training example.
    """
    #import pdb; pdb.set_trace()
    image = tf.cast(example['inputs'], dtype=dtype)
    if subset != "test":
        label = example['label'] #tf.one_hot(example['label'], 17035) if onehot_label else example['label']
    if data_augmentations is not None:
        if 'glc_default' in data_augmentations:
            if subset == "train":
                #example["label"] = tf.one_hot(example["label"], 17035)
                image = dataset_utils.augment_random_crop_flip(
                    image, crop_size, crop_size, num_channels, crop_padding=0, flip=True)
                image = tf.cast(image, dtype=dtype)
                return {'inputs': image, 'label': label}
		#abel = tf.one_hot(exanple['label']) if onehot_labels else label
                #return {'inputs': image, 'label': label}
            else:
                #print("-----IMAGE VALIDATION--------")
                #print(tf.shape(example["inputs"]))
                #image = tf.image.resize_with_crop_or_pad(
                #    image, crop_size, crop_size)
                image = dataset_utils.augment_random_crop_flip(image, crop_size,crop_size, num_channels,crop_padding=0, flip=False)
                image = tf.cast(image, dtype=dtype)
                if subset == "validation":
                    #example["label"] = tf.one_hot(example["label"], 17035)
                    return {'inputs': image, 'label': label}
                else:
                    return {'inputs': image}
    if subset != "test":
        return {'inputs': image, 'label': label}
    else:
        return {'inputs': image}


@datasets.add_dataset('glc_dataset')
def get_dataset(
    *,
    batch_size: int,
    eval_batch_size: int,
    bands,
    num_shards,
    dtype_str: Text = 'float32',
    shuffle_seed: Optional[int] = 0,
    rng: Optional[Rng] = None,
    prefetch_buffer_size=2,
    dataset_configs: ml_collections.ConfigDict,
        dataset_service_address: Optional[str] = None) -> dataset_utils.Dataset:
    """Returns a generator for the dataset."""
    del rng  # Parameter was required by caller API, but is unused.

    def validate_config(field):
        if dataset_configs.get(field) is None:
            raise ValueError(
                f'{field} must be specified for TFRecord dataset.')

    validate_config('base_dir')
    validate_config('tables')
    validate_config('examples_per_subset')
    validate_config('num_classes')

    onehot_labels = dataset_configs.get('onehot_labels', True)
    crop_size = dataset_configs.get('crop_size', 224)
    data_augmentations = dataset_configs.get('data_augmentations', None)
    num_channels = get_num_channels(bands)
    data_aug_train = functools.partial(get_augmentations, onehot_label=onehot_labels, num_channels=num_channels,
                                       dtype=tf.float32, data_augmentations=data_augmentations, crop_size=crop_size, subset="train")
    data_aug_val = functools.partial(get_augmentations, onehot_label=onehot_labels, num_channels=num_channels,
                                     dtype=tf.float32, data_augmentations=data_augmentations, crop_size=crop_size, subset="validation")
    data_aug_test = functools.partial(get_augmentations, onehot_label=onehot_labels, num_channels=num_channels,
                                      dtype=tf.float32, data_augmentations=data_augmentations, crop_size=crop_size, subset="test")
    #augmentation_params = dataset_configs.get('augmentation_params', None)
   # data_augmentations = functools(partial(get_augmentations,augmentation_params)

    #onehot_labels = dataset_configs.get('onehot_labels', True)
    # For the test set, the actual batch size is
    # test_batch_size * num_test_clips.
    test_batch_size = dataset_configs.get('test_batch_size', eval_batch_size)
    test_split = dataset_configs.get('test_split', 'test')

    ds_factory = functools.partial(
        TFRecordDatasetFactory,
        base_dir=dataset_configs.base_dir,
        tables=dataset_configs.tables,
        examples_per_subset=dataset_configs.examples_per_subset,
        num_classes=dataset_configs.num_classes,
    )

    def create_dataset_iterator(

        subset: Text,
        batch_size_local: int,
        transform=None,
    ) -> Tuple[Iterator[Batch], int]:
        is_training = subset == 'train'
        is_test = subset == 'test'
        logging.info('Loading split %s', subset)

        dataset, num_examples = glc_load_split(
            ds_factory,
            batch_size,
            subset,
            bands=bands,
            dtype=tf.float32,
            image_size=crop_size,
            shuffle_seed=None,
            transform=transform)

        if dataset_service_address and is_training:
            if shuffle_seed is not None:
                raise ValueError('Using dataset service with a random seed causes each '
                                 'worker to produce exactly the same data. Add '
                                 'config.shuffle_seed = None to your config if you '
                                 'want to run with dataset service.')
            logging.info('Using the tf.data service at %s',
                         dataset_service_address)
            dataset = dataset_utils.distribute(
                dataset, dataset_service_address)

        current_ds_iterator = (
            dataset_utils.tf_to_numpy(data) for data in iter(dataset)
        )

        return current_ds_iterator, num_examples

    shard_batches = functools.partial(
        dataset_utils.shard, n_devices=num_shards)
    train_iter, n_train_examples = create_dataset_iterator(
        'train', batch_size, transform=data_aug_train)
    train_iter = map(shard_batches, train_iter)
#    train_iter = jax_utils.prefetch_to_device(train_iter, prefetch_buffer_size)

    eval_iter, n_eval_examples = create_dataset_iterator(
        'validation', eval_batch_size, transform=data_aug_val)
    eval_iter = map(shard_batches, eval_iter)
   # eval_iter = jax_utils.prefetch_to_device(eval_iter, prefetch_buffer_size)

    test_iter, n_test_examples = create_dataset_iterator(
        'test', test_batch_size, transform=data_aug_test)  # create_dataset_iterator(
    # test_split, test_batch_size) #, transform=data_aug)
    test_iter = map(shard_batches, test_iter)
    #test_iter = jax_utils.prefetch_to_device(test_iter, prefetch_buffer_size)

    meta_data = {
        'num_classes': dataset_configs.num_classes,
        'input_shape': (-1, crop_size, crop_size, num_channels),
        'num_train_examples': n_train_examples,
        'num_eval_examples': n_eval_examples,
        'num_test_examples':
        n_test_examples,
        'input_dtype': getattr(jnp, dtype_str),
        'target_is_onehot': onehot_labels,
    }
    logging.info('Dataset metadata:\n%s', meta_data)

    return dataset_utils.Dataset(train_iter, eval_iter, test_iter, meta_data)
