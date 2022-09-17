import os

import gin
import numpy as np

from objective_functions.tf_objective_functions import FUNCTIONS
from environments.create_environment import create_environment
from evaluation.evaluation_utils import build_eval_params
import tensorflow as tf

from evaluation.plot_utils import plot, plot_performance_over_time_with_stds, plot_performance_by_function


@gin.configurable
class EvaluationDriver:
    def __init__(
            self,
            run_dir,
            num_observations,
            environment_type,

            # optional gin parameters
            train_episode_length=50,
            episode_length=200,
            input_dimension=2,
            number_optimization_parameters=2,
            plot_all=True):

        self.dtype = tf.float32
        self.run_dir = run_dir
        self.train_episode_length = train_episode_length

        self.n_start_pos = 3
        self.starting_positions = build_eval_params(self.n_start_pos, input_dimension)
        self.plot_all = plot_all
        self.batch_size = len(self.starting_positions)

        self.envs = []
        for label, functions in FUNCTIONS.items():
            self.envs.append(create_environment(
                environment_type,
                label,
                (functions[0],),
                self.starting_positions,
                episode_length,
                num_observations,
                self.batch_size,
                input_dimension,
                number_optimization_parameters
            ))

    def run(self, policy, step_counter, log_summary=False):
        plot_dir = os.path.join(self.run_dir, "Step_{}".format(step_counter))
        performances = []
        names = []
        mean_performance_over_time = []
        std_performance_over_time = []
        for env in self.envs:
            rewards = []
            time_step = env.reset()
            rewards.append(time_step.reward)
            policy_state = policy.get_initial_state(env.batch_size)

            while not tf.reduce_all(time_step.is_last()):
                action_step = policy.action(time_step, policy_state=policy_state)
                policy_state = action_step.state
                time_step = env.step(action_step.action)
                rewards.append(time_step.reward)

            performance, means, stds = plot(
                step_counter,
                plot_dir,
                self.n_start_pos,
                function_values=tf.transpose(FUNCTIONS["ackley"][0](tf.transpose(env.get_states()))),
                states=env.get_states(),
                name=env.name,
                train_episode_length=self.train_episode_length,
                log_summary=log_summary
            )

            env.reset()

            performances.append(performance)
            mean_performance_over_time.append(tf.reduce_mean(means, axis=0))
            std_performance_over_time.append(tf.reduce_mean(stds, axis=0))
            names.append(env.name)
            if not log_summary:
                print("\r{} evaluated".format(", ".join(names)), end="")
        if log_summary:
            print("\n")
        else:
            print("\r", end="")
        run_id = os.path.split(self.run_dir)[1]
        train_function_name = os.path.split(os.path.split(self.run_dir)[0])[1]
        algorithm_name = os.path.split(os.path.split(os.path.split(self.run_dir)[0])[0])[1]
        plot_performance_over_time_with_stds(range(self.envs[0].episode_length),
                                             np.array(mean_performance_over_time),
                                             np.array(std_performance_over_time),
                                             FUNCTIONS.keys(),
                                             "{}-{}-{}-agent"
                                             .format(run_id, algorithm_name, train_function_name),
                                             plot_dir,
                                             "performance over time",
                                             std_scale=0.25)

        plot_performance_over_time_with_stds(range(self.train_episode_length),
                                             np.array(mean_performance_over_time)[:, :self.train_episode_length],
                                             np.array(std_performance_over_time)[:, :self.train_episode_length],
                                             FUNCTIONS.keys(),
                                             "{}-{}-{}-agent"
                                             .format(run_id, algorithm_name, train_function_name),
                                             plot_dir,
                                             "performance over time {} steps".format(self.train_episode_length),
                                             std_scale=0.25)

        plot_performance_by_function(FUNCTIONS.keys(), performances, plot_dir, "final performance")
        plot_performance_by_function(FUNCTIONS.keys(),
                                     np.array(mean_performance_over_time)[:, self.train_episode_length - 1],
                                     plot_dir, "final performance at {} steps".format(self.train_episode_length))

        summary = ["average performances by function at step {} \n".format(step_counter)]
        for label, performance in zip(FUNCTIONS.keys(), performances):
            summary.append("{} function performance = {:.2f} \n".format(label, performance))

        f = open(os.path.join(plot_dir, "summary.txt"), 'w')
        f.writelines(summary)

        return performances
