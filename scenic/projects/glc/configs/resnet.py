# pylint: disable=line-too-long
r"""Default configs for ResNet on ImageNet.

"""
# pylint: enable=line-too-long

import ml_collections

GLC_TRAIN_SIZE = 1587395


def get_config():
    """Returns the base experiment configuration for ImageNet."""
    config = ml_collections.ConfigDict()
    config.experiment_name = 'glc_resnet'
    # Dataset.
    config.dataset_name = 'glc_dataset'

    config.data_dtype_str = 'float32'
    config.dataset_configs = ml_collections.ConfigDict()

    config.base_dir = (
      #'/network/scratch/t/tengmeli/temp_glc/')
            "/mnt/disks/persist/" )
            #'/home/tengmeli/')
    config.tables = {
      'train': 'train_image.tfrecords',
      'validation': 'val_image.tfrecords',
      'test': 'test_image.tfrecords'
    }
    config.examples_per_subset = {
      'train': 1587395,
      'validation': 40080,
      'test': 36421
    }
    
    config.no_comet = False
    config.comet= {"tags":["test_run"]}
 
    config.onehot_labels= False   
    config.bands = ["rgb", "near_ir"] #, "near_ir"]
    config.num_classes = 17035
    config.crop_size=224
    config.data_augmentations = ["glc_default"]
    # Model.
    config.model_name = 'resnet_classification'
    config.num_filters = 64
    config.num_layers = 50
    config.model_dtype_str = 'float32'

    # Training.
    config.trainer_name = 'classification_trainer'
    config.optimizer = 'momentum'
    config.optimizer_configs = ml_collections.ConfigDict()
    config.optimizer_configs.momentum = 0.9
    config.l2_decay_factor = .00005
    config.max_grad_norm = None
    config.label_smoothing = None
    config.num_training_epochs = 90
    config.batch_size = 64 #32 #8192
    config.rng_seed = 0
    config.init_head_bias = -10.0
    
    config.num_shards=4    
    # Learning rate.
    steps_per_epoch = GLC_TRAIN_SIZE // config.batch_size
    total_steps = config.num_training_epochs * steps_per_epoch
    base_lr = 0.1 * config.batch_size / 256
    # setting 'steps_per_cycle' to total_steps basically means non-cycling cosine.
    config.lr_configs = ml_collections.ConfigDict()
    config.lr_configs.learning_rate_schedule = 'compound'
    config.lr_configs.factors = 'constant * cosine_decay * linear_warmup'
    config.lr_configs.warmup_steps = 7 * steps_per_epoch
    config.lr_configs.steps_per_cycle = total_steps
    config.lr_configs.base_learning_rate = base_lr

    # Logging.
    config.write_summary = True
    config.xprof = True  # Profile using xprof.
    config.checkpoint = True  # Do checkpointing.
    config.checkpoint_steps = 10 * steps_per_epoch
    config.debug_train = True  # Debug mode during training.
    config.debug_eval = False  # Debug mode during eval.
    return config


