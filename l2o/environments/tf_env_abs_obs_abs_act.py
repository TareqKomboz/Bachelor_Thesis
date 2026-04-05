import tensorflow as tf
from tf_agents.utils import common

from l2o.environments.tf_env import TfEnv


class TfEnvAbsObsAbsAct(TfEnv):
    def __init__(
            self,
            input_dimension,
            function_name,
            objective_function,
            number_free_parameters,
            starting_position,
            batch_size,
            episode_length):
        super(TfEnvAbsObsAbsAct, self).__init__(
            input_dimension=input_dimension,
            function_name=function_name,
            objective_function=objective_function,
            number_free_parameters=number_free_parameters,
            starting_position=starting_position,
            batch_size=batch_size,
            episode_length=episode_length
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
        expanded_action = clipped_action  # why not: tf.expand_dims(clipped_action, axis=0)
        self._state.assign(tf.concat(
            values=[self.free_values, expanded_action],
            axis=1
        ))
