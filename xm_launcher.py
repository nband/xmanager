# coding=utf-8
# Copyright 2021 The Uncertainty Baselines Authors.
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

"""Uncertainty Baselines launcher."""

import collections
import functools
import getpass
import importlib.util
import json
import operator
import os
import random
import shutil
import tempfile
import time
import asyncio
from typing import Any, Dict, List, Optional, Text

from absl import app
from absl import flags
from absl import logging
from ml_collections.config_dict import config_dict
# pylint: disable=g-import-not-at-top
try:
  from xmanager import xm as xm_oss
  from xmanager import xm_local
  from xmanager.contrib import copybara
  from xmanager.contrib import tpu as xm_oss_tpu
  from xmanager.cloud import build_image
  from xmanager.cloud import utils
  from xmanager.cloud import caip
except (ImportError, ModuleNotFoundError):
  logging.exception('Cannot import open-sourced XM.')
  xm_oss = None
  xm_local = None
  copybara = None
  xm_oss_tpu = None

try:
  from uncertainty_baselines import halton
except (ImportError, ModuleNotFoundError):
  logging.exception('Cannot import halton sequence generator.')
  halton = None

hyper = None
xm = None

# pylint: enable=g-import-not-at-top

# Binary flags
flags.DEFINE_string(
    'binary',
    None,
    'Filepath to Python script to run. For external GCS experiments, it can be '
    'an absolute path to the binary, or a relative one with respect to the '
    'current folder.'
)
flags.mark_flag_as_required('binary')
flags.DEFINE_list(
    'args', [], 'Flag arguments to pass to binary. Follow the format '
    '--args=batch_size=64,train_epochs=300.')
flags.DEFINE_string(
    'config', None, 'Filepath to Python file with a function '
    'get_sweep(hyper) returning a hyperparameter sweep and/or '
    'a function get_config() returning a ConfigDict.')
flags.DEFINE_bool(
    'use_halton_generator', False,
    'Whether to use the open-sourced Halton generator or an internal generator '
    'to generate hyperparameter sweeps.')
flags.DEFINE_bool('launch_on_gcp', False, 'Whether or not to launch on GCS.')
flags.DEFINE_string(
    'cell',
    None,
    'Cloud region or cell for the worker (and coordinator if using TPU).')

# Accelerator flags
flags.DEFINE_string('platform', None, 'Platform (e.g., tpu-v2, tpu-v3, gpu).')
flags.DEFINE_string(
    'tpu_topology',
    '2x2',
    'TPU topology. Only used if platform is TPU. {x}x{y} means x*x **chips**, '
    'and because the number of devices is the number of cores, we further '
    'multiply by 2 because there are 2 cores per chip. For example, 2x2 is '
    'equivalent to an 8 core TPU slice, 8x8 = 128 cores, etc.')
flags.DEFINE_string('gpu_type', 'p100',
                    'GPU type. Only used if platform is GPU.')
flags.DEFINE_integer('num_gpus', None,
                     'Number of GPUs. Only used if platform is GPU.')
flags.DEFINE_integer('num_cpus', None,
                     'Number of CPUs. Only used if launching on GCP.')
flags.DEFINE_integer(
    'memory', None, 'Amount of CPU memory in GB. Only used if launching on '
    'GCP.')
flags.DEFINE_string('experiment_name', None,
                    'Experiment name; defaults to timestamp.')
flags.DEFINE_integer('num_runs', 1,
                     'Number of runs each with a different seed.')
flags.DEFINE_string('tensorboard', None, 'Tensorboard instance.')


FLAGS = flags.FLAGS


_JobMetadata = collections.namedtuple('_JobMetadata', [
    'user', 'cell', 'platform_str', 'gpu_type', 'num_gpus', 'tpu_topology',
    'num_cpus', 'experiment_name', 'memory',
])


def _get_attr(config, name: str) -> Optional[Any]:
  """Get a given attribute from the passed FLAGS or ConfigDict."""
  # Note that if a flag is passed with its default value, this will not override
  # a conflicting config value.
  has_flag_value = name in FLAGS and FLAGS[name].value != FLAGS[name].default
  if has_flag_value:
    return FLAGS[name].value
  elif config and name in config:
    return config[name]
  elif name in FLAGS:
    return FLAGS[name].default
  return None


def _build_binary_metadata(config):
  """Extracts job metadata and args from the given ConfigDict and/or FLAGS."""
  if FLAGS.binary[:2] == '//':
    # We assume the path will have at least two cmds split by '/' and
    # We will use the last two to name the experiment.
    # Ideally, the path will look like //.../{dataset}/{baseline}.py
    # but {dataset} and {baseline} can be any string in practice.
    command = FLAGS.binary.split('/')
    if len(command) >= 2:
      dataset = command[-2]
      baseline = command[-1]
      baseline = os.path.splitext(baseline)[0]
    else:
      dataset = None
      baseline = None
  else:
    pieces = FLAGS.binary.split('/')
    dataset = pieces[-2]
    baseline = pieces[-1]
    baseline = os.path.splitext(baseline)[0]
  if config:
    flag_args = config.args
    experiment_name = _get_attr(config, 'experiment_name')
  else:
    flag_args = dict(arg.split('=', 1) for arg in FLAGS.args)
    experiment_name = FLAGS.experiment_name
  dataset = flag_args.get('dataset', dataset)

  if not experiment_name:  # default experiment name
    experiment_name = time.strftime('%m%d_%H%M%S')
    if baseline is not None:
      experiment_name = f'{baseline}-{experiment_name}'
    if dataset is not None:
      experiment_name = f'{dataset}-{experiment_name}'
    if not experiment_name.islower():
      experiment_name = f'ub-{experiment_name}'

  user = _get_attr(config, 'user')
  metadata = _JobMetadata(
      user=user,
      cell=_get_attr(config, 'cell'),
      platform_str=_get_attr(config, 'platform'),
      gpu_type=_get_attr(config, 'gpu_type'),
      num_gpus=_get_attr(config, 'num_gpus'),
      tpu_topology=_get_attr(config, 'tpu_topology'),
      num_cpus=_get_attr(config, 'num_cpus'),
      memory=_get_attr(config, 'memory'),
      experiment_name=experiment_name,
  )


  # use_gpu = 'gpu' in metadata.platform_str or metadata.platform_str == 'cpu'

  if metadata.platform_str == 'cpu':
    num_cores = 1
  elif 'gpu' in metadata.platform_str:
    num_cores = metadata.num_gpus
  else:
    num_cores = 2 * functools.reduce(
        operator.mul, [int(i) for i in metadata.tpu_topology.split('x')])
  if 'num_cores' in flag_args and flag_args['num_cores'] != num_cores:
    raise ValueError(
        '"num_cores" requested in binary incompatible with inferred number of '
        'cores based on tpu_topology and platform_str ({}!={} respectively)'
        .format(flag_args['num_cores'], num_cores))
  args = dict(num_cores=num_cores,
              # use_gpu=use_gpu
              )
  args.update(flag_args)
  return args, metadata


def _split_path_to_ub(filepath):
  """For a path '/a/b/c/baselines/...', return '/a/b/c', 'baselines/...'."""
  filepath = os.path.abspath(filepath)
  pieces = filepath.split('/')
  dir_index = None
  for pi, piece in enumerate(pieces):
    if piece in ['experimental', 'baselines']:
      dir_index = pi
      break
  if dir_index is None:
    raise ValueError(
        'Unable to parse FLAGS.binary ({}) to find the location of the '
        'uncertainty_baselines project.'.format(filepath))
  project_dir = '/'.join(pieces[:dir_index])
  binary_path = '/'.join(pieces[dir_index:])
  return project_dir, binary_path


def _launch_gcp_experiment(project_dir, binary_path, sweep, args, metadata):
  """Launch a job on GCP using the Cloud AI Platform."""
  logging.info('Using %s as the project dir.', project_dir)

  # TODO(znado): support different caip regions, etc.?
  with xm_local.create_experiment(metadata.experiment_name) as experiment:
    # Note that we normally would need to append a "$@" in order to properly
    # forward the args passed to the job into the python command, but the XM
    # library already does this for us.
    run_cmd = f'python {binary_path}'
    # These images are necessary to get tf-nightly pre-installed.
    platform_docker_instructions = []
    if 'tpu' in metadata.platform_str:
      # tpuvm requires Python3.8 and GLIBC_2.29, which requires at least
      # debian:11 or ubuntu:20.04.
      base_image = 'ubuntu:20.04'
      platform_docker_instructions = (
          ['RUN apt-get install -y python-is-python3 python3-pip wget'] +
          list(xm_oss_tpu.tpuvm_docker_instructions()))
    elif metadata.platform_str == 'gpu':
      base_image = 'tensorflow/tensorflow:nightly-gpu'
    else:
      base_image = 'tensorflow/tensorflow:nightly'
    pip_cmd = 'pip --no-cache-dir install'
    spec = xm_oss.PythonContainer(
        path=project_dir,
        base_image=base_image,
        entrypoint=xm_oss.CommandList([run_cmd]),
        # docker_instructions=[
        #     f'COPY {os.path.basename(project_dir)}/ uncertainty-baselines',
        #     'RUN apt-get update && apt-get install -y git netcat',
        #     ] + platform_docker_instructions + [
        #     'RUN python -m pip install --upgrade pip setuptools wheel',
        #     f'RUN {pip_cmd} google-cloud-storage',
        #     # f'RUN {pip_cmd} ./uncertainty-baselines[experimental,models]',
        #     # f'RUN {pip_cmd} ./uncertainty-baselines[experimental,models,datasets,jax,torch]',
        #     f'RUN {pip_cmd} ./uncertainty-baselines[experimental,models,datasets,torch]',  # Try excluding jax
        #   'WORKDIR uncertainty-baselines',
        # ],
        docker_instructions=[
          f'COPY {os.path.basename(project_dir)}/ uncertainty-baselines',
          'RUN apt-get update && apt-get install -y git netcat',
        ] + platform_docker_instructions + [
          'RUN python -m pip install --upgrade pip setuptools wheel',
          f'RUN {pip_cmd} google-cloud-storage',
          f'RUN {pip_cmd} ./uncertainty-baselines[jax,models]',
          f'RUN {pip_cmd} wandb torch seaborn dm-haiku',
          'WORKDIR uncertainty-baselines',
        ],
    )
    [executable] = experiment.package([
        xm_oss.Packageable(
            executable_spec=spec,
            executor_spec=xm_local.Caip.Spec(),
        ),
    ])

    # [executable] = experiment.package([
    #     xm_oss.Packageable(
    #         executable_spec=spec,
    #         executor_spec=xm_local.Caip.Spec(),
    #         args={
    #             # TODO: replace workerpool0 with the actual name of
    #             # the job when uCAIP supports custom name worker pools.
    #             'master_addr_port':
    #                 xm_oss.ShellSafeArg(
    #                     utils.get_workerpool_address('workerpool0')),
    #         },
    #     ),
    # ])

    platform = {}
    # num_cpus = metadata.num_cpus
    # memory = metadata.memory
    target_num_cpus = 16
    # target_num_cpus = 33
    # target_num_cpus = 31
    target_memory = 60
    # target_memory = 60  # n1-standard-16, this works
    # target_memory = 104  # n1-highmem-16, this doesn't work
    # target_memory = 119

    # High mem 32?
    # target_num_cpus = 32
    # target_memory = 208

    # High mem 64
    # target_num_cpus, target_memory = 64, 416

    # High mem 96
    # target_num_cpus, target_memory = 96, 624

    # Somehow 33, 209 -> n1-standard-64, with (64, 240), this works
    # target_num_cpus, target_memory = 33, 209

    # And 31, 119 -> n1-standard-32
    # target_num_cpus, target_memory = 31, 119

    # High mem 2
    # target_num_cpus, target_memory = 1, 4

    # Try for a2 - highgpu - 4g
    # target_num_cpus, target_memory = 48, 340

    num_cpus = (
      min(metadata.num_cpus, target_num_cpus)
      if metadata.num_cpus else target_num_cpus)
    memory = (
      min(metadata.memory, target_memory)
      if metadata.memory else target_memory)

    if 'tpu' in metadata.platform_str:
      pieces = list(map(int, metadata.tpu_topology.split('x')))
      num_tpus = pieces[0] * pieces[1] * 2  # 2 cores per TPU chip.
      # Rename tpu-v3 -> tpu_v3.
      platform = {metadata.platform_str.replace('-', '_'): num_tpus}
      args['tpu'] = 'local'
    elif metadata.platform_str == 'gpu':
      platform = {metadata.gpu_type: metadata.num_gpus}

    if num_cpus is not None:
      # platform['cpu'] = num_cpus * xm_oss.vCPU
      platform['cpu'] = num_cpus
    if memory is not None:
      platform['memory'] = memory * xm_oss.GiB
      # platform['memory'] = memory
    # executor = xm_local.Caip(requirements=xm_oss.JobRequirements(**platform))

    # Create one job per setting in the hyperparameter sweep. The default case
    # is a length 1 sweep with a single argument name "seed".
    # job_group_args = {}

    tensorboard = FLAGS.tensorboard
    if not tensorboard:
      tensorboard = caip.client().create_tensorboard('diabetic_retinopathy')
      tensorboard = asyncio.get_event_loop().run_until_complete(tensorboard)

    for ji, sweep_args in enumerate(sweep):
      job_args = args.copy()
      if 'output_dir' in job_args:
        job_args['output_dir'] = os.path.join(job_args['output_dir'], str(ji))
      if 'data_dir' in job_args and job_args.get('download_data', False):
        job_args['data_dir'] = os.path.join(job_args['data_dir'], str(ji))
      # Overwrite any values in `args` with the `sweep_args`.
      job_args.update(sweep_args)

      tensorboard_capability = xm_local.TensorboardCapability(
          name=tensorboard, base_output_directory=job_args['output_dir'])
      logging.info(
          'Launching job %d/%d with args %s.\n',
          ji + 1,
          len(sweep),
          json.dumps(job_args, indent=4, sort_keys=True))
      # job = xm_oss.Job(
      #     executable=executable,
      #     executor=executor,
      #     args=job_args,
      # )
      job = xm_oss.Job(
        executable=executable,
        executor=xm_local.Caip(
          requirements=xm_oss.JobRequirements(**platform),
          tensorboard=tensorboard_capability),
        args=job_args)
      # job_group_args[str(ji)] = job

      experiment.add(job)

    # job_group = xm_oss.JobGroup(**job_group_args)
    # experiment.add(job=job_group)


def _generate_hyperparameter_sweep(
    config_module,
    config: config_dict.ConfigDict,
    project_dir: Optional[str]) -> List[Dict[Text, Any]]:
  """Generate the hyperparameter sweep."""
  config_use_halton_generator = (
      FLAGS.config and
      'use_halton_generator' in config and
      config.use_halton_generator)
  use_halton_generator = (
      FLAGS.use_halton_generator or config_use_halton_generator or not hyper)
  hyper_module = halton if use_halton_generator else hyper
  if use_halton_generator and halton is None:
    # We should only need to do this on external GCS.
    halton_path = os.path.join(project_dir, 'uncertainty_baselines/halton.py')
    hyper_module_spec = importlib.util.spec_from_file_location(
        '', os.path.abspath(halton_path))
    hyper_module = importlib.util.module_from_spec(hyper_module_spec)
    hyper_module_spec.loader.exec_module(hyper_module)
    print('loaded hyper module: ', hyper_module)
  if FLAGS.config and 'get_sweep' in dir(config_module):
    if hyper_module is None:
      raise ValueError('Need a hyperparameter module to construct sweep.')
    if FLAGS.num_runs != 1:
      raise ValueError('FLAGS.num_runs not supported with config.get_sweep().')
    sweep = config_module.get_sweep(hyper_module)
  else:
    sweep = [
        {'seed': seed + random.randint(0, 1e10)}
        for seed in range(FLAGS.num_runs)
    ]
  return sweep


def _load_config_helper(config_path):
  config_module_spec = importlib.util.spec_from_file_location(
      '', os.path.abspath(config_path))
  config_module = importlib.util.module_from_spec(config_module_spec)
  config_module_spec.loader.exec_module(config_module)
  config = None
  if 'get_config' in dir(config_module):
    config = config_module.get_config()
  return config_module, config


def _load_config(config_path):
  """Load the ConfigDict if one was passed in as FLAGS.config."""
  if config_path:
    config_module = None
    if not config_module:
      config_module, config = _load_config_helper(config_path)
  else:
    config_module = None
    config = None
  return config_module, config


def main(argv):
  del argv  # unused arg
  config_module, config = _load_config(FLAGS.config)
  args, metadata = _build_binary_metadata(config)
  if FLAGS.launch_on_gcp:
    project_dir, binary_path = _split_path_to_ub(FLAGS.binary)
    sweep = _generate_hyperparameter_sweep(config_module, config, project_dir)
    return _launch_gcp_experiment(
        project_dir, binary_path, sweep, args, metadata)


if __name__ == '__main__':
  app.run(main)
