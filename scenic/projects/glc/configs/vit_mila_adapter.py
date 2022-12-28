# pylint: disable=line-too-long
r"""Default configs for Regularized ViT on ImageNet2012.

Based on: https://arxiv.org/pdf/2106.10270.pdf

"""
# pylint: disable=line-too-long

import ml_collections

_IMAGENET_TRAIN_SIZE = 1281167
NUM_CLASSES = 1000
GLC_TRAIN_SIZE = 40080
VARIANT = 'B/16'


def get_config(runlocal=''):
  """Returns the ViT experiment configuration for ImageNet."""

  runlocal = bool(runlocal)

  config = ml_collections.ConfigDict()
  config.experiment_name = 'glc_vit_adapter'
  # Dataset.
  config.dataset_name = 'glc_dataset'
  config.data_dtype_str = 'float32'
  config.dataset_configs = ml_collections.ConfigDict()
  config.dataset_configs.dataset = "glc_dataset"#'imagenet2012'
  config.no_comet = True
  config.dataset_configs = ml_collections.ConfigDict()

  config.base_dir = (
      '/network/scratch/t/tengmeli/temp_glc/')
  config.tables = {
      'train': 'val_images_new3.tfrecords',
      'validation': 'val_images_new3.tfrecords',
      'test': 'test_images_new.tfrecords'
    }
  config.examples_per_subset = {
      'train': 40080,#1587395,
      'validation': 40080,
      'test': 36421
    }
  config.onehot_labels= False   
  config.bands = ["rgb", "near_ir"] #, "near_ir"]
  config.num_classes = 17035
  config.crop_size=224
  config.data_augmentations = ["glc_default"]

  config.dataset_configs.pp_train = (
      'decode_jpeg_and_inception_crop(224)|flip_lr'
      '|randaug(2, 15)'
      '|value_range(-1, 1)'
      f'|onehot({NUM_CLASSES}, key="label", key_result="labels")'
      '|keep("image", "labels")')
  config.dataset_configs.pp_eval = (
      'decode'
      '|resize_small(256)|central_crop(224)'
      '|value_range(-1, 1)'
      f'|onehot({NUM_CLASSES}, key="label", key_result="labels")'
      '|keep("image", "labels")')
  config.dataset_configs.prefetch_to_device = 2
  # Shuffle_buffer_size is per host, so small-ish is ok.
  config.dataset_configs.shuffle_buffer_size = 250_000

  # Model.
  version, patch = VARIANT.split('/')
  config.model_name = 'vit_adapter_classification'
  config.model = ml_collections.ConfigDict()

  config.model.hidden_size = {'Ti': 192,
                              'S': 384,
                              'B': 768,
                              'L': 1024,
                              'H': 1280}[version]
  config.model.patches = ml_collections.ConfigDict()
  config.model.patches.size = [int(patch), int(patch)]
  config.model.num_heads = {'Ti': 3, 'S': 6, 'B': 12, 'L': 16, 'H': 16}[version]
  config.model.mlp_dim = {'Ti': 768,
                          'S': 1536,
                          'B': 3072,
                          'L': 4096,
                          'H': 5120}[version]
  config.model.num_layers = {'Ti': 12,
                             'S': 12,
                             'B': 12,
                             'L': 24,
                             'H': 32}[version]
  config.model.representation_size = None
  config.model.classifier = 'token'
  config.model.attention_dropout_rate = 0.0
  config.model.dropout_rate = 0.1
  config.model.stochastic_depth = 0.1
  config.model_dtype_str = 'float32'

  # Training.
  config.trainer_name = 'classification_trainer'
  config.optimizer = 'adam'
  config.optimizer_configs = ml_collections.ConfigDict()
  config.optimizer_configs.freeze_params_reg_exp = "(Transformer.encoderblock*|Transformer.pos_embed*|Transformer.encoder*|init_bn|embedding*)"
  config.optimizer_configs.beta1 = 0.9
  config.optimizer_configs.beta2 = 0.999
  config.optimizer_configs.weight_decay = 0.1
  config.explicit_weight_decay = None  # No explicit weight decay
  config.l2_decay_factor = None
  config.max_grad_norm = 1.0
  config.label_smoothing = None
  config.num_training_epochs = 300
  config.log_eval_steps = 1000
  config.batch_size = 1 if runlocal else 1 #4096
  config.rng_seed = 42
  config.init_head_bias = -6.9  # -log(1000)

  # Learning rate.
  steps_per_epoch = _IMAGENET_TRAIN_SIZE // config.batch_size
  total_steps = config.num_training_epochs * steps_per_epoch
  base_lr = 0.001
  config.lr_configs = ml_collections.ConfigDict()
  config.lr_configs.learning_rate_schedule = 'compound'
  config.lr_configs.factors = 'constant * cosine_decay * linear_warmup'
  config.lr_configs.warmup_steps = 10_000
  config.lr_configs.steps_per_cycle = total_steps
  config.lr_configs.base_learning_rate = base_lr

  # Mixup.
  config.mixup = ml_collections.ConfigDict()
  config.mixup.bind_to = None
  config.mixup.alpha = 0.5

  # Logging.
  config.write_summary = True
  config.xprof = True  # Profile using xprof.
  config.checkpoint = True  # Do checkpointing.
  config.checkpoint_steps = 5000
  config.debug_train = False  # Debug mode during training.
  config.debug_eval = False  # Debug mode during eval.

  config.m = None  # Placeholder for randaug strength.
  config.l = None  # Placeholder for randaug layers.
  config.adapter_layers=""
  config.adapter_dim=32
  config.models_default_hparams = {
      # Default num_classes has no effect since it is always overwritten or used
      # for rand init models whose head is always replaced.
      'num_classes': 1,
      # Set to ids_ints2str(range(max_num_layers)) to activate all adapters.
      'adapter_layers': '',
      'num_layers': 3,
      'adapter_dim': 32,
      'opt_lr': 0.01,
      'opt_lr_schedule': 'cosine',
      'opt_lr_warmup_ratio': 0.1,
      'opt_momentum': 0.9,
      'opt_nesterov': False,
      'ds_image_size': 32,
      'ds_area_range_min': 0.05,
      'ds_aspect_ratio_range_min': 0.75,
      'ds_flip_left_right': True,
      'ds_brightness_delta': 0.0,
      'ds_contrast_delta': 0.0,
      'ds_saturation_delta': 0.0,
      'ds_hue_delta': 0.0,
  }


  return config


