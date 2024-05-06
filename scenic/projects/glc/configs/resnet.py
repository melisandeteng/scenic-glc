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
      '/network/scratch/t/tengmeli/temp_glc/')


    config.tables = {
      'train': 'train_images_new6.tfrecords',
      'validation':'val_images_new6.tfrecords',
      'test': 'test_images_new5.tfrecords'
    }
    config.examples_per_subset = {
      'train':1587395,
      'validation':40080,
      'test': 36421
    }
    
    config.no_comet = True
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
    config.trainer_name = 'transfer_trainer'
    config.optimizer = 'momentum'
    config.optimizer_configs = ml_collections.ConfigDict()
    config.optimizer_configs.momentum = 0.9
    config.optimizer_configs.skip_scale_and_bias_regularization=True
    config.l2_decay_factor = .00005
    config.max_grad_norm = None
    config.skip_scale_and_bias_regularization=True
    config.label_smoothing = None
    config.num_training_epochs = 90
    config.batch_size =  128 #8192
    config.rng_seed = 0
    config.init_head_bias = -10.0
    
    config.num_shards=4    
    # Learning rate.
    steps_per_epoch = GLC_TRAIN_SIZE // config.batch_size
    total_steps = config.num_training_epochs * steps_per_epoch
    base_lr = 0.01 #* config.batch_size / 256
    # setting 'steps_per_cycle' to total_steps basically means non-cycling cosine.
    config.lr_configs = ml_collections.ConfigDict()
    config.lr_configs.learning_rate_schedule = 'compound'
    config.lr_configs.factors = "constant"#'constant * cosine_decay * linear_warmup'

    #config.lr_configs.decay_events =  [0]
    #config.lr_configs.decay_factors = [0.01]
    config.lr_configs.warmup_steps = 0 #7 * steps_per_epoch
    config.lr_configs.steps_per_cycle = total_steps
    config.lr_configs.base_learning_rate = 0.01 #base_lr
    config.init_from = {'checkpoint_path': "/mnt/disks/persist/checkpoints", "model_config":"/home/tengmeli/scenic-glc/scenic/projects/baselines/configs/imagenet/imagenet_resnet_config.py"}
    # Logging.
    config.write_summary = True
    config.xprof = True  # Profile using xprof.
    config.checkpoint = True  # Do checkpointing.
    config.checkpoint_steps = 10 * steps_per_epoch
    config.debug_train = True  # Debug mode during training.
    config.debug_eval = False  # Debug mode during eval.
    config.init_from = {'checkpoint_path': "/network/projects/_groups/ecosystem-embeddings/scenic_glc/resnet_ckpt/", "model_config":"/home/mila/t/tengmeli/scenic-glc/scenic/projects/baselines/configs/imagenet/imagenet_resnet_config.py"}
    return config


