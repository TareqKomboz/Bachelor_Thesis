import tensorflow as tf
from tf_agents.utils import common

from environments.tf_env import TfEnv


class TfEnvAbsObsAbsAct(TfEnv):
    def __init__(
            self,
            name,
            objective_function,
            starting_position,
            batch_size):
        super(TfEnvAbsObsAbsAct, self).__init__(
            name=name,
            objective_function=objective_function,
            starting_position=starting_position,
            batch_size=batch_size
        )

    def _reset_observations(self):
        self._observations[:, 0].assign(tf.concat((
            self._initial_state,
            tf.expand_dims(self._function_values[:, 0], axis=1)),
            axis=1
        ))

    def _build_observation(self):
        self._observations[:, self._steps].assign(tf.concat((
            self._state,
            tf.expand_dims(self._function_values[:, self._steps], axis=1)),
            axis=1
        ))

    def _assign_state(self, action):
        clipped_action = common.clip_to_spec(value=action, spec=self.action_spec())
        expanded_action = clipped_action  # clipped_action # tf.expand_dims(clipped_action, axis=0)
        self._state.assign(tf.concat(
            values=[self.free_values, expanded_action],
            axis=1
        ))

        # state = common.clip_to_spec(action, self.action_spec())
        # self._state.assign(state)
