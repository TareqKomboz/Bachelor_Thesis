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
    def __init__(
            self,
            name,
            objective_functions,
            starting_position,
            episode_length,
            num_observations,
            batch_size,
            input_dimension,
            number_optimization_parameters):
        self.name = name
        self.objective_functions = objective_functions
        if len(objective_functions) == 1:
            self.evaluate_objective_function = self._evaluate_objective_function
        else:
            self.evaluate_objective_function = self._evaluate_objective_functions
        self._initial_state = starting_position
        self.episode_length = tf.constant(episode_length, dtype=tf.int64)
        self._num_observations = tf.constant(num_observations, dtype=tf.int64)
        self._batch_size = batch_size
        self.input_dimension = tf.constant(input_dimension, dtype=tf.int64)
        self.number_optimization_parameters = number_optimization_parameters
        self.number_free_parameters = input_dimension - number_optimization_parameters

        self._dtype = tf.float32
        self._steps = common.create_variable('step', 0)
        self._resets = common.create_variable('resets', 0)

        self.observation_shape = self.input_dimension + 1
        observation_spec = specs.BoundedTensorSpec(
            [num_observations, self.observation_shape],
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

    # Todo: FREE 2: Tareq
    # def initialize_input(self):
    #     # initial values of optimization parameters -> in each step() new value(s) for optimization parameters
    #     initial_state = array((random.uniform(
    #         low=-1.0,
    #         high=1.0,
    #         size=self.number_optimization_parameters
    #     )), dtype=float32)
    #
    #     # random values for non optimization parameters -> fixed value(s) until next reset()
    #     random_parameter_values = array(random.uniform(
    #         low=-1.0,
    #         high=1.0,
    #         size=self.number_free_parameters
    #     ), dtype=float32)
    #
    #     return current_action, random_parameter_values

    def _current_time_step(self):
        reward = self._function_values[:, self._steps]

        if tf.less(self._steps, self._num_observations):
            observation = tf.pad(
                self._observations[:, self._steps:: -1],
                [[0, 0], [0, self._num_observations - self._steps - 1], [0, 0]]
            )
        else:
            observation = self._observations[:, self._steps:self._steps - self._num_observations:-1]

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
        reward = self.objective_functions[0](tf.transpose(x))
        return reward

    def _evaluate_objective_functions(self, x):
        x = tf.reshape(
            x,
            (len(self.objective_functions), int(self.batch_size / len(self.objective_functions)), self.input_dimension)
        )
        reward = []
        for i in range(len(x)):
            reward.append(self.objective_functions[i](tf.transpose(x[i])))
        reward = tf.reshape(reward, (self.batch_size,))
        return reward

    def set_starting_positions(self, starting_positions):
        self._initial_state = starting_positions

    def get_states(self):
        return self._states
