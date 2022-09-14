import tensorflow as tf
from tf_agents.utils import common

from environments.tf_env_2d import TfEnv2d


class TfEnv2DAbsObsAbsAct(TfEnv2d):
    def __init__(self,
                 name,
                 obj_fcts,
                 starting_position,
                 episode_length,
                 num_observations,
                 batch_size):
        super(TfEnv2DAbsObsAbsAct, self).__init__(name,
                                                  obj_fcts,
                                                  starting_position,
                                                  episode_length,
                                                  num_observations,
                                                  batch_size)

    def _reset_observations(self):
        self._observations[:, 0].assign(tf.concat((self._initial_state,
                                                   tf.expand_dims(self._function_values[:, 0], axis=1)),
                                                  axis=1))

    def _build_observation(self):
        self._observations[:, self._steps].assign(
            tf.concat((self._state,
                       tf.expand_dims(self._function_values[:, self._steps], axis=1))
                      , axis=1))

    def _assign_state(self, action):
        state = common.clip_to_spec(action, self.action_spec())
        self._state.assign(state)
