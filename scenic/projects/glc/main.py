# Copyright 2022 The Scenic Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Main file for Scenic."""
import comet_ml
from absl import flags
from absl import logging
from clu import metric_writers
from flax.training import checkpoints
import jax
import jax.numpy as jnp
import ml_collections
from scenic import app
from scenic.model_lib import models
from scenic.train_lib import train_utils
from scenic.train_lib import trainers
import scenic.projects.glc.data.glc_dataset as glc_data
import os 

from comet_ml import ExistingExperiment, Experiment
from time import sleep
import resource
from pathlib import Path
#from ray.train._internal.utils import get_address_and_port
low, high = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (high, high))
#os.environ['CUDA_VISIBLE_DEVICES'] = '0, 1'
#address, port = get_address_and_port()
#init_ip=address+":"+ str(port)
#jax.distributed.initialize(init_ip, num_processes=2, process_id=0)
#jax.distributed.initialize(init_ip, num_processes=2, process_id=1)
devices = jax.local_devices()
comet_kwargs = {
    "auto_metric_logging": False,
    "parse_args": True,
    "log_env_gpu": True,
    "log_env_cpu": True,
    "display_summary_level": 0,
}

print(devices)
#os.environ["XLA_FLAGS"]="--xla_gpu_cuda_data_dir=/cvmfs/ai.mila.quebec/apps/arch/common/cuda/11.2"
#FLAGS = flags.FLAGS

#jax.distributed.initialize()
    
def main(rng: jnp.ndarray, config: ml_collections.ConfigDict, workdir: str,
         writer: metric_writers.MetricWriter) -> None:
  """Main function for Scenic."""
  exp=None
  if not config.no_comet:
        # ----------------------------------
        # -----  Set Comet Experiment  -----
        # ----------------------------------

    print("Starting new experiment")
    exp = Experiment(project_name="scenic", **comet_kwargs)
    if config.comet.tags:
        config.comet.tags = list(config.comet.tags)
        print("Logging to comet.ml with tags",  config.comet.tags)
        exp.add_tags( config.comet.tags)

    
    exp.log_asset_folder(
                str(Path(__file__).parent / "scenic"),
                recursive=True,
                log_file_name=True,
            )
    exp.log_asset(str(Path(__file__)))
    
    exp.log_parameters(config)
    sleep(1)
    
    print("Running model in", workdir)

  #import pdb; pdb.set_trace()
  model_cls = models.get_model_cls(config.model_name)
  data_rng, rng = jax.random.split(rng)
  bands = config.bands
  
  if config.checkpoint:
    # When restoring from a checkpoint, change the dataset seed to ensure that
    # the example order is new. With deterministic data, this ensures enough
    # randomization and in the future with deterministic data + random access,
    # we can feed the global step to the dataset loader to always continue
    # reading the rest of the data if we resume a job that was interrupted.
    checkpoint_path = checkpoints.latest_checkpoint(workdir)
    logging.info('CHECKPOINT PATH: %s', checkpoint_path)
    if checkpoint_path is not None:
      global_step = train_utils.checkpoint_path_step(checkpoint_path) or 0
      logging.info('Folding global_step %s into dataset seed.', global_step)
      data_rng = jax.random.fold_in(data_rng, global_step)
  #(TFDS_DATA_DIR)
  dataset = glc_data.get_dataset(
      dataset_configs=config,
    batch_size=config.batch_size,
    eval_batch_size=config.batch_size,
    num_shards=1,
    dtype_str = 'float32',
    bands = bands)

  trainers.get_trainer(config.trainer_name)(
      rng=rng,
      config=config,
      model_cls=model_cls,
      dataset=dataset,
      workdir=workdir,
      writer=writer,
      comet_exp=exp)


if __name__ == '__main__':
  app.run(main=main)



    

