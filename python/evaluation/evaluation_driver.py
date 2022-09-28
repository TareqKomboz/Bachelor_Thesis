import os

import gin

from objective_functions.tf_objective_functions import FUNCTIONS
from environments.create_environment import create_environment
from evaluation.evaluation_utils import build_evaluation_parameters
import tensorflow as tf

from evaluation.plot_utils import plot, plot_performance_over_time_with_stds


@gin.configurable("evaluation_driver_init")
class EvaluationDriver:
    def __init__(
            self,
            run_dir,
            environment_type,
            input_dimension,
            function_name,
            number_free_parameters,
            episode_length,

            # optional gin parameters
            n_start_pos,
            plot_all=True):

        self.run_dir = run_dir
        self.n_start_pos = n_start_pos

        self.starting_positions = build_evaluation_parameters(n_start_pos=self.n_start_pos, input_dimension=input_dimension)  # todo: not as batchsize
        self.plot_all = plot_all
        self.batch_size = len(self.starting_positions)

        self.environment = create_environment(
            environment_type=environment_type,
            input_dimension=input_dimension,
            function_name=function_name,
            objective_function=FUNCTIONS[function_name],
            number_free_parameters=number_free_parameters,
            start_point=self.starting_positions,
            batch_size=self.batch_size,
            episode_length=episode_length
        )

    def run(self, policy, step_counter):
        plot_dir = os.path.join(self.run_dir, "Step_{}".format(step_counter))

        time_step = self.environment.reset()
        policy_state = policy.get_initial_state(self.environment.batch_size)
        while not tf.reduce_all(time_step.is_last()):
            action_step = policy.action(time_step, policy_state=policy_state)
            policy_state = action_step.state
            time_step = self.environment.step(action_step.action)

        (
            average_reward_over_batches_and_actions,
            reward_stds_over_batches_and_actions,
            average_return_over_batch,
            return_stds_over_batch,
            average_final_objective_function_value_over_batch,
            final_objective_function_value_stds_over_batch,
            average_max_reward_of_episode_over_batches,
            max_reward_of_episode_stds_over_batches,
            reward_means_over_batch,
            reward_stds_over_batch
        ) = plot(
            step_counter=step_counter,
            plot_dir=plot_dir,
            function_values=self.environment.get_function_values()
        )

        self.environment.reset()

        print("\r", end="")

        plot_performance_over_time_with_stds(
            x=range(self.environment.get_episode_length()),
            means=reward_means_over_batch,
            stds=reward_stds_over_batch,
            title="{}D {} - {} free - Convergence".format(
                self.environment.get_input_dimension(),
                self.environment.get_function_name(),
                self.environment.get_number_free_parameters()
            ),
            plot_dir=plot_dir,
            filename="convergence",
            std_scale=0.25
        )

        return [
            average_reward_over_batches_and_actions,
            reward_stds_over_batches_and_actions,
            average_return_over_batch,
            return_stds_over_batch,
            average_final_objective_function_value_over_batch,
            final_objective_function_value_stds_over_batch,
            average_max_reward_of_episode_over_batches,
            max_reward_of_episode_stds_over_batches
        ]
