import gin
import tensorflow as tf
from tf_agents import specs
from tf_agents.trajectories import time_step as ts
from tf_agents.utils import common
from tf_agents.environments import TFEnvironment

from numpy import ones

FIRST = ts.StepType.FIRST
MID = ts.StepType.MID
LAST = ts.StepType.LAST


class TfEnv(TFEnvironment):
    @gin.configurable("environment_constructor")
    def __init__(
            self,
            input_dimension,
            function_name,
            objective_function,
            number_free_parameters,
            starting_position,
            batch_size,
            episode_length,

            # gin
            number_observations,
            randomize_start):

        self.function_name = function_name
        self.objective_function = objective_function
        self.evaluate_objective_function = self._evaluate_objective_function
        self._initial_state = starting_position
        self._batch_size = batch_size
        self.episode_length = tf.constant(episode_length, dtype=tf.int64)
        self._number_observations = tf.constant(number_observations, dtype=tf.int64)
        self.input_dimension = tf.constant(input_dimension, dtype=tf.int64)
        self.number_free_parameters = number_free_parameters
        self.randomize_start = randomize_start
        self.number_optimization_parameters = self.input_dimension - self.number_free_parameters

        self._dtype = tf.float32
        self._steps = common.create_variable('step', 0)
        self._resets = common.create_variable('resets', 0)

        self.free_values = self._initial_state[:, :self.number_free_parameters]

        self.observation_shape = self.input_dimension + 1
        observation_spec = specs.BoundedTensorSpec(
            [number_observations, self.observation_shape],
            self._dtype,
            minimum=tuple((-1 * ones((self.observation_shape,), dtype=int)).tolist()),
            maximum=tuple((ones((self.observation_shape,), dtype=int)).tolist())
        )

        self.action_shape = (self.number_optimization_parameters,)
        action_spec = specs.BoundedTensorSpec(
            self.action_shape,
            self._dtype,
            minimum=tuple((-1 * ones(self.action_shape, dtype=int)).tolist()),
            maximum=tuple((ones(self.action_shape, dtype=int)).tolist())
        )
        reward_spec = specs.BoundedTensorSpec((), self._dtype, minimum=-0, maximum=1)
        discount_spec = specs.BoundedTensorSpec((), self._dtype, minimum=0, maximum=1)
        step_type_spec = specs.BoundedTensorSpec((), tf.int32, minimum=0, maximum=1)
        time_step_spec = ts.TimeStep(
            step_type=step_type_spec,
            reward=reward_spec,
            discount=discount_spec,
            observation=observation_spec
        )
        super(TfEnv, self).__init__(time_step_spec, action_spec, batch_size=batch_size)

        self._state = common.create_variable(
            'state',
            self._initial_state,
            shape=(batch_size, self.input_dimension),
            dtype=self._dtype
        )
        self._states = common.create_variable(
            'states',
            shape=(batch_size, episode_length, self.input_dimension),
            dtype=self._dtype
        )
        self._function_values = common.create_variable(
            'function_values',
            shape=(batch_size, episode_length,),
            dtype=self._dtype
        )
        self._observations = common.create_variable(
            'observations',
            shape=(batch_size, episode_length, self.observation_shape),
            dtype=self._dtype
        )

    def _current_time_step(self):
        reward = self._function_values[:, self._steps]

        if tf.less(self._steps, self._number_observations):
            observation = tf.pad(
                self._observations[:, self._steps:: -1],
                [[0, 0], [0, self._number_observations - self._steps - 1], [0, 0]]
            )
        else:
            observation = self._observations[:, self._steps:self._steps - self._number_observations:-1]

        def first():
            return tf.constant(FIRST, dtype=tf.int32)

        def mid():
            return tf.constant(MID, dtype=tf.int32)

        def last():
            return tf.constant(LAST, dtype=tf.int32)

        step_type = tf.case(
            [(tf.equal(self._steps, 0), first), (tf.equal(self._steps, self.episode_length - 1), last)],
            default=mid
        )

        step_type = tf.tile((step_type,), (self.batch_size,), name='step_type')
        return ts.TimeStep(step_type, reward, tf.ones(shape=(self._batch_size,)), observation)

    def _reset(self):
        self._steps.assign(0)
        self._resets.assign_add(1)
        # todo: if self.randomize_start:
        #    self.set_starting_positions_and_free_values()
        self._state.assign(self._initial_state)
        self._states.assign(tf.zeros_like(self._states))
        self._states[:, 0].assign(self._state)
        self._function_values.assign(tf.zeros_like(self._function_values))
        self._function_values[:, 0].assign(self.evaluate_objective_function(self._initial_state))
        self._observations.assign(tf.zeros_like(self._observations))
        self._reset_observations()

        return self.current_time_step()

    def _step(self, action):
        self._steps.assign_add(1)
        if self._steps >= self.episode_length:
            return self.reset()
        self.assign_state(action)
        self._states[:, self._steps].assign(self._state)
        self._function_values[:, self._steps].assign(self.evaluate_objective_function(self._state))
        self.build_observation()
        return self.current_time_step()

    def render(self):
        self.plot(None)

    def build_observation(self):
        self._build_observation()

    def assign_state(self, action):
        self._assign_state(action)

    def _evaluate_objective_function(self, x):
        reward = self.objective_function(tf.transpose(x))
        return reward

    # def _evaluate_objective_functions(self, x):
    #     number_objective_functions = len(self.objective_functions)
    #     x = tf.reshape(
    #         x,
    #         (number_objective_functions, int(self.batch_size / number_objective_functions), self.input_dimension)
    #     )
    #     reward = []
    #     for i in range(number_objective_functions):
    #         reward.append(self.objective_functions[i](tf.transpose(x[i])))
    #     reward = tf.reshape(reward, (self.batch_size,))
    #     return reward

    def set_starting_positions_and_free_values(self):
        start_point = tf.random.uniform(
            shape=(self.batch_size, self.input_dimension),
            minval=tuple((-1 * ones((self.input_dimension,), dtype=int)).tolist()),
            maxval=tuple((ones((self.input_dimension,), dtype=int)).tolist()),
            dtype=tf.float32
        )

        self._initial_state = start_point
        self.free_values = self._initial_state[:, :self.number_free_parameters]

    def get_states(self):
        return self._states

    def get_episode_length(self):
        return self.episode_length

    def get_input_dimension(self):
        return self.input_dimension

    def get_function_name(self):
        return self.function_name

    def get_function_values(self):
        return self._function_values

    def get_number_free_parameters(self):
        return self.number_free_parameters
