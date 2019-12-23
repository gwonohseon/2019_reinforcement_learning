from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="1"
from absl import app
from absl import flags
from absl import logging

import numpy as np
import tensorflow as tf

from tf_agents.agents.categorical_dqn import categorical_dqn_agent
from tf_agents.environments import batched_py_environment
import suite_retro
import retro
from tf_agents.eval import metric_utils
from tf_agents.metrics import py_metric
from tf_agents.metrics import py_metrics
from tf_agents.networks import categorical_q_network
from tf_agents.policies import epsilon_greedy_policy
from tf_agents.policies import py_tf_policy
from tf_agents.policies import random_py_policy
from tf_agents.replay_buffers import py_hashed_replay_buffer
from tf_agents.specs import tensor_spec
from tf_agents.trajectories import policy_step
from tf_agents.trajectories import time_step as ts
from tf_agents.trajectories import trajectory
from tf_agents.utils import common
from tf_agents.utils import timer

'''
Load the check point.
Run the game.
'''

tf.compat.v1.enable_v2_behavior()

root_dir = './logs/sonic_cdqn'


class AtariCategoricalQNetwork(categorical_q_network.CategoricalQNetwork):
  """CategoricalQNetwork subclass that divides observations by 255."""

  def call(self, observation, step_type=None, network_state=None):
    state = tf.cast(observation, tf.float32)
    # We divide the grayscale pixel values by 255 here rather than storing
    # normalized values beause uint8s are 4x cheaper to store than float32s.
    # TODO(b/129805821): handle the division by 255 for train_eval_atari.py in
    # a preprocessing layer instead.
    state = state / 255
    return super(AtariCategoricalQNetwork, self).call(
        state, step_type=step_type, network_state=network_state)


conv_layer_params=((32, (8, 8), 4), (64, (4, 4), 2), (64, (3, 3), 1))
fc_layer_params=(512,)
gamma=0.99
eval_epsilon_greedy=0.

root_dir = os.path.expanduser(root_dir)
train_dir = os.path.join(root_dir, 'train')

non_batch_env = suite_retro.load('SonicTheHedgehog-Genesis', max_episode_steps=108000//4, terminal_on_life_loss=False)


_env = batched_py_environment.BatchedPyEnvironment([non_batch_env])

observation_spec = tensor_spec.from_spec(_env.observation_spec())
time_step_spec = ts.time_step_spec(observation_spec)
action_spec = tensor_spec.from_spec(_env.action_spec())

optimizer = tf.compat.v1.train.RMSPropOptimizer(
            learning_rate=2.5e-4,
            decay=0.95,
            momentum=0.0,
            epsilon=0.00001,
            centered=True)
categorical_q_net = AtariCategoricalQNetwork(
    observation_spec,
    action_spec,
    conv_layer_params=conv_layer_params,
    fc_layer_params=fc_layer_params)
agent = categorical_dqn_agent.CategoricalDqnAgent(
    time_step_spec,
    action_spec,
    categorical_q_network=categorical_q_net,
    epsilon_greedy=0.01,
    n_step_update=2,
    optimizer=optimizer,
    gamma=gamma)


_train_checkpointer = common.Checkpointer(
            ckpt_dir=train_dir,
            agent=agent)

_policy_checkpointer = common.Checkpointer(
            ckpt_dir=os.path.join(train_dir, 'policy'),
            policy=agent.policy)

_train_checkpointer.initialize_or_restore()
_policy_checkpointer.initialize_or_restore()

_eval_policy = epsilon_greedy_policy.EpsilonGreedyPolicy(
                policy=agent.policy,
                epsilon=eval_epsilon_greedy)

_eval_policy.action = common.function(_eval_policy.action)


import cv2

for _ in range(100):
    time_step = _env.reset()
    a = _env.envs[0].render(mode='rgb_array')
    cv2.imshow('render', a)
    cv2.waitKey(10)
    i = 0
    while not time_step.is_last():
      i += 1
      # print(i)
      action_step = _eval_policy.action(time_step) # _eval_policy.action(time_step) # agent.policy.action(time_step)
      # action_step = agent.policy.action(time_step)
      time_step = _env.step(action_step.action)
      # print(time_step.reward)
      a =_env.envs[0].render(mode='rgb_array')
      cv2.imshow('render', a)
      cv2.waitKey(10)
