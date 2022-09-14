import tensorflow as tf

from tf_agents import specs
from tf_agents.trajectories import time_step as ts
from tf_agents.utils import common

from tf_agents.environments import TFEnvironment

from common.utils import create_rotation_matrix

FIRST = ts.StepType.FIRST
MID = ts.StepType.MID
LAST = ts.StepType.LAST


class TfEnv2d(TFEnvironment):
    def __init__(self,
                 name,
                 obj_fcts,
                 starting_position,
                 episode_length,
                 num_observations,
                 batch_size):
        self.name = name
        self.obj_fcts = obj_fcts
        if len(obj_fcts) == 1:
            self.evaluate_obj_fct = self._evaluate_obj_fct
        else:
            self.evaluate_obj_fct = self._evaluate_obj_fcts

        self._batch_size = batch_size
        self._dtype = tf.float32
        observation_spec = specs.BoundedTensorSpec([num_observations, 3], self._dtype, minimum=(-1, -1, -1),
                                                   maximum=(1, 1, 1))
        action_spec = specs.BoundedTensorSpec((2,), self._dtype, minimum=(-1, -1), maximum=(1, 1))
        reward_spec = specs.BoundedTensorSpec((), self._dtype, minimum=-0, maximum=1)
        discount_spec = specs.BoundedTensorSpec((), self._dtype, minimum=0, maximum=1)
        step_type_spec = specs.BoundedTensorSpec((), tf.int32, minimum=0, maximum=1)
        time_step_spec = ts.TimeStep(step_type=step_type_spec,
                                     reward=reward_spec,
                                     discount=discount_spec,
                                     observation=observation_spec)
        super(TfEnv2d, self).__init__(time_step_spec, action_spec, batch_size=batch_size)
        self._initial_state = starting_position
        self._state = common.create_variable('state', self._initial_state, shape=(batch_size, 2), dtype=self._dtype)
        self._states = common.create_variable('states', shape=(batch_size, episode_length, 2), dtype=self._dtype)
        self._function_values = common.create_variable('function_values',
                                                       shape=(batch_size, episode_length,), dtype=self._dtype)
        self._observations = common.create_variable('observations', shape=(batch_size, episode_length, 3),
                                                    dtype=self._dtype)
        self._steps = common.create_variable('step', 0)
        self._resets = common.create_variable('resets', 0)
        self.episode_length = tf.constant(episode_length, dtype=tf.int64)
        self._num_observations = tf.constant(num_observations, dtype=tf.int64)

    def _current_time_step(self):
        reward = self._function_values[:, self._steps]

        if tf.less(self._steps, self._num_observations):
            observation = tf.pad(self._observations[:, self._steps:: -1],
                                 [[0, 0], [0, self._num_observations - self._steps - 1], [0, 0]])
        else:
            observation = self._observations[:, self._steps:self._steps - self._num_observations:-1]

        def first():
            return tf.constant(FIRST, dtype=tf.int32)

        def mid():
            return tf.constant(MID, dtype=tf.int32)

        def last():
            return tf.constant(LAST, dtype=tf.int32)

        step_type = tf.case(
            [(tf.equal(self._steps, 0), first),
             (tf.equal(self._steps, self.episode_length - 1), last)],
            default=mid)

        step_type = tf.tile((step_type,), (self.batch_size,), name='step_type')
        return ts.TimeStep(step_type, reward, tf.ones(shape=(self._batch_size,)), observation)

    def _reset(self):
        self._steps.assign(0)
        self._resets.assign_add(1)
        self._state.assign(self._initial_state)
        self._states.assign(tf.zeros_like(self._states))
        self._states[:, 0].assign(self._state)
        self._function_values.assign(tf.zeros_like(self._function_values))
        self._function_values[:, 0].assign(self.evaluate_obj_fct(self._initial_state))
        self._observations.assign(tf.zeros_like(self._observations))
        self._reset_observations()

        return self.current_time_step()

    def _step(self, action):
        self._steps.assign_add(1)
        if self._steps >= self.episode_length:
            return self.reset()
        self.assign_state(action)
        self._states[:, self._steps].assign(self._state)
        self._function_values[:, self._steps].assign(
            self.evaluate_obj_fct(self._state))
        self.build_observation()
        return self.current_time_step()

    def render(self):
        self.plot(None)

    def build_observation(self):
        self._build_observation()

    def assign_state(self, action):
        self._assign_state(action)

    def _evaluate_obj_fct(self, X):
        reward = self.obj_fcts[0](tf.transpose(X))
        return reward

    def _evaluate_obj_fcts(self, X):
        X = tf.reshape(X, (len(self.obj_fcts), int(self.batch_size / len(self.obj_fcts)), 2))
        reward = []
        for i in range(len(X)):
            reward.append(self.obj_fcts[i](tf.transpose(X[i])))
        reward = tf.reshape(reward, (self.batch_size,))
        reward += tf.random.normal([1], mean=0, stddev=self._output_noise, dtype=self._dtype)[:, 0]
        return reward

    def set_starting_positions(self, starting_positions):
        self._initial_state = starting_positions

    def get_states(self):
        return self._states

    def get_function_values(self):
        return self._function_values
