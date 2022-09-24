import os

import gin
import numpy as np

from objective_functions.tf_objective_functions import FUNCTIONS
from environments.create_environment import create_environment
from evaluation.evaluation_utils import build_evaluation_params
import tensorflow as tf

from evaluation.plot_utils import plot, plot_performance_over_time_with_stds, plot_performance_by_function


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

        self.starting_positions = build_evaluation_params(n_start_pos=self.n_start_pos, input_dimension=input_dimension)
        self.plot_all = plot_all
        self.batch_size = len(self.starting_positions)

        self.environment = create_environment(
            environment_type=environment_type,
            input_dimension=input_dimension,
            function_name=function_name,
            objective_function=FUNCTIONS[function_name][0],
            number_free_parameters=number_free_parameters,
            start_point=self.starting_positions,
            batch_size=self.batch_size,
            episode_length=episode_length
        )

    def run(self, policy, step_counter, log_summary=False):
        plot_dir = os.path.join(self.run_dir, "Step_{}".format(step_counter))
        average_final_objective_function_values = []
        names = []
        mean_performance_over_time = []
        std_performance_over_time = []




        time_step = self.environment.reset()
        policy_state = policy.get_initial_state(self.environment.batch_size)

        while not tf.reduce_all(time_step.is_last()):
            action_step = policy.action(time_step, policy_state=policy_state)
            policy_state = action_step.state
            time_step = self.environment.step(action_step.action)

        average_final_objective_function_value_over_batch, means, stds = plot(
            step_counter=step_counter,
            plot_dir=plot_dir,
            function_values=self.environment.get_function_values(),
            name=self.environment.name,
            log_summary=log_summary
        )

        self.environment.reset()

        average_final_objective_function_values.append(average_final_objective_function_value_over_batch)
        mean_performance_over_time.append(tf.reduce_mean(means, axis=0))
        std_performance_over_time.append(tf.reduce_mean(stds, axis=0))
        names.append(self.environment.name)
        if not log_summary:
            print("\r{} evaluated".format(", ".join(names)), end="")




        if log_summary:
            print("\n")
        else:
            print("\r", end="")
        run_id = os.path.split(self.run_dir)[1]
        train_function_name = os.path.split(os.path.split(self.run_dir)[0])[1]
        algorithm_name = os.path.split(os.path.split(os.path.split(self.run_dir)[0])[0])[1]
        plot_performance_over_time_with_stds(
            x=range(self.environment.get_episode_length()),
            means=np.array(mean_performance_over_time),
            stds=np.array(std_performance_over_time),
            labels=FUNCTIONS.keys(),
            title="{}-{}-{}-agent".format(run_id, algorithm_name, train_function_name),
            plot_dir=plot_dir,
            filename="performance over time",
            std_scale=0.25
        )

        summary = ["average performances by function at step {} \n".format(step_counter)]
        for label, average_final_objective_function_value in zip(FUNCTIONS.keys(), average_final_objective_function_values):
            summary.append("{} function average_final_objective_function_value = {:.2f} \n".format(label, average_final_objective_function_value))

        f = open(os.path.join(plot_dir, "summary.txt"), 'w')
        f.writelines(summary)

        return average_final_objective_function_values
