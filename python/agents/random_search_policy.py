import tensorflow as tf
from tf_agents import specs
from tf_agents.policies import TFPolicy
from tf_agents.trajectories import time_step as ts, policy_step
from tf_agents.typing import types


class RandomSearchPolicy(TFPolicy):
    def __init__(self, time_step_spec: ts.TimeStep,
                 action_spec: types.NestedTensorSpec,
                 batch_size,
                 episode_length,
                 input_dimension,
                 number_optimization_parameters,
                 info_spec: types.NestedTensorSpec = (),):
        policy_state_spec = specs.BoundedTensorSpec((4,), tf.float32, minimum=(0, -1, -1, -1), maximum=(1e38, 1, 1, 1))
        super().__init__(time_step_spec, action_spec, policy_state_spec, info_spec)
        self.episode_length = tf.constant(episode_length, tf.float32)
        self.step = tf.Variable(name="step", initial_value=0, shape=(), dtype=tf.int32)
        self.max = tf.Variable(name="max", initial_value=tf.zeros((batch_size, 3)), shape=(batch_size, 3),
                               dtype=tf.float32)
        self.batch_size = batch_size
        self.input_dimension = input_dimension
        self.number_optimization_parameters = number_optimization_parameters

    def _action(self, time_step: ts.TimeStep, policy_state: types.NestedTensorSpec, **kwargs) -> policy_step.PolicyStep:
        reward = time_step.reward
        counter = policy_state[:, 0] + 1
        max_coords = policy_state[:, 1:3]
        max_reward = policy_state[:, 3]

        is_normal = counter < self.episode_length - 1
        is_normal_mask = tf.cast(is_normal, tf.float32)
        normal_action = tf.random.uniform([self.batch_size, self.number_optimization_parameters], minval=-1, maxval=1) \
                        * tf.stack((is_normal_mask, is_normal_mask), 1)

        is_last = counter >= self.episode_length - 1
        is_last_mask = tf.cast(is_last, tf.float32)
        last_action = max_coords * tf.stack((is_last_mask, is_last_mask), 1)
        action = normal_action + last_action

        greater = reward > max_reward
        greater_mask = tf.cast(greater, tf.float32)
        new_state = time_step.observation[:, 0] * tf.stack((greater_mask, greater_mask, greater_mask), 1)
        loe = reward <= max_reward
        loe_mask = tf.cast(loe, tf.float32)
        old_state = policy_state[:, 1:] * tf.stack((loe_mask, loe_mask, loe_mask), 1)

        state = tf.concat((tf.expand_dims(counter, axis=1), old_state + new_state), axis=1)
        info = ()

        return policy_step.PolicyStep(action, state, info)

    def _distribution(
            self, time_step: ts.TimeStep,
            policy_state: types.NestedTensorSpec) -> policy_step.PolicyStep:
        return policy_step.PolicyStep((), (), ())
