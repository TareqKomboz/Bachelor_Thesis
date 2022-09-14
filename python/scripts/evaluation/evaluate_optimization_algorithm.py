import argparse
import math
import os
import time

import numpy as np
import tensorflow as tf
from scipy import optimize

from objective_functions.tf_objective_functions import FUNCTIONS
from common.utils import create_rotation_matrix
from definitons import RUNS_DIR
from evaluation.evaluation_utils import build_eval_params
from evaluation.plot_utils import plot_performance_over_time_with_stds, plot, plot_performance_by_function
from db.runs import save_to_sql

episode_length = 200


class NpFuncWrapper:
    def __init__(self, func,
                 translation=0,
                 rotation=0,
                 input_noise=0,
                 output_noise=0,
                 invert=False):
        self.func = func
        self.translation = translation
        self.rotation = create_rotation_matrix(rotation)
        self.input_noise = input_noise
        self.output_noise = output_noise
        self.invert = invert
        self.max = -1.0
        self.trajectory = []

    def evaluate(self, X):
        X = tf.convert_to_tensor(X, tf.float32)
        X = tf.linalg.matvec(self.rotation, X)
        X = tf.subtract(X, self.translation)
        X += tf.random.normal(X.shape, mean=0.0, stddev=self.input_noise, dtype=tf.float32)
        reward = self.func(tf.transpose(X))
        reward += tf.random.normal(reward.shape, mean=0.0, stddev=self.output_noise, dtype=tf.float32)
        if self.invert:
            reward = -reward
        return np.array(reward, np.float64)

    def evaluate_with_tracking(self, X):
        out = self.evaluate(X)

        value = -out[0] if self.invert else out[0]
        if value > self.max:
            self.max = value

        self.trajectory.append(np.array((X[0], X[1], value, self.max)))
        return out

    def reset_tracking(self):
        trajectory = self.trajectory.copy()
        self.trajectory = []
        return trajectory


def run(method_name, file, file2):
    n_start_pos = 3
    n_trans = 4
    N_rotations = 10
    N_input_noise = 10
    N_output_noise = 10

    domain = tf.constant([[-1, -1], [1, 1]], dtype=tf.float32)
    starting_positions, translations, rotations, input_noises, output_noises = build_eval_params(
        n_start_pos,
        n_trans,
        N_rotations,
        N_input_noise,
        N_output_noise,
        domain)
    batch_size = len(starting_positions)

    start = time.time()
    mean_performance_over_time = []
    std_performance_over_time = []
    for function_name in FUNCTIONS.keys():
        performances = []
        fcts = NpFuncWrapper(FUNCTIONS[function_name][0],
                             translation=translations,
                             rotation=rotations,
                             input_noise=input_noises,
                             output_noise=output_noises).evaluate
        if method_name == "powell" or method_name == "nelder-mead":
            trajectories = []
            for i in range(batch_size):
                timestamp = time.strftime('%H:%M:%S', time.gmtime(time.time() - start))
                print("{} ({}): {}-{} minimization, {}"
                      .format(i + 1, batch_size, method_name, function_name, timestamp), end="")
                wrapper = NpFuncWrapper(FUNCTIONS[function_name][0],
                                        translation=translations[i],
                                        rotation=rotations[i],
                                        input_noise=input_noises[i],
                                        output_noise=output_noises[i],
                                        invert=True)
                fct = wrapper.evaluate_with_tracking
                optimize.minimize(fct,
                                  starting_positions[i],
                                  method=method_name,
                                  bounds=domain.numpy().T,
                                  options={'return_all': False,
                                           'maxiter': episode_length})
                trajectories.append(wrapper.reset_tracking())
                print(end="\r")
            lengths = np.array([len(trajectory) for trajectory in trajectories])
            max_length = max(np.max(lengths), episode_length)

            padded_trajectories = np.array([np.pad(np.array(trajectory),
                                                   ((0, max_length - len(trajectory)),
                                                    (0, 0)),
                                                   'edge')
                                            for trajectory in trajectories])
            states = padded_trajectories[:, :episode_length, :2]
            function_values = padded_trajectories[:, :episode_length, 2]
            timestamp = time.strftime('%H:%M:%S', time.gmtime(time.time() - start))
            print(
                "{} ({}): {}-{} minimizations finished, avg performance={:.2f}"
                " mean number of evaluations: {:.2f}, max={}, min={}, std: {:.2f}"
                ", {}"
                    .format(i + 1, batch_size, method_name, function_name, np.mean(function_values[:, -1]),
                            np.mean(lengths), np.max(lengths), np.min(lengths), np.std(lengths),
                            timestamp))

        elif method_name == "random_search":
            np.random.seed(0)
            actions = np.random.uniform(-1, 1, (episode_length, batch_size, 2))
            maxs = np.zeros((episode_length, batch_size))
            max_locations = actions.copy()
            rewards = np.zeros_like(maxs)
            for i in range(episode_length):
                rewards[i] = fcts(actions[i])[:, 0]
                maxs[i] = np.where(rewards[i] > maxs[i - 1], rewards[i], maxs[i - 1])
                max_locations[i] = np.array(
                    (np.where(rewards[i] > maxs[i - 1], actions[i, :, 0], max_locations[i - 1, :, 0]),
                     np.where(rewards[i] > maxs[i - 1], actions[i, :, 1], max_locations[i - 1, :, 1]))).T
            states = tf.convert_to_tensor(np.moveaxis(actions, (0, 1, 2), (1, 0, 2)), tf.float32)
            function_values = tf.convert_to_tensor(fcts(max_locations), tf.float32)
            timestamp = time.strftime('%H:%M:%S', time.gmtime(time.time() - start))
            print("{} - {} minimization finished, {}"
                  .format(method_name, function_name, timestamp))
        elif method_name == "grid_search":
            n = int(math.sqrt(episode_length))
            X = tf.range(domain[0][0] + 0.05, domain[1][0], (domain[1][0] - domain[0][0] - 0.1) / (n - 1))
            Y = tf.range(domain[0][1] + 0.05, domain[1][1], (domain[1][1] - domain[0][1] - 0.1) / (n - 1))
            actions = np.reshape(np.meshgrid(X, Y), (2, n ** 2)).T
            actions = np.pad(actions, ((0, episode_length - len(actions)), (0, 0)), 'edge')
            actions = np.expand_dims(actions, axis=0)
            actions = np.tile(actions, (batch_size, 1, 1))

            maxs = np.zeros((episode_length, batch_size))
            max_locations = actions.copy()
            rewards = np.zeros_like(maxs)
            for i in range(episode_length):
                rewards[i] = fcts(actions[:, i])[:, 0]
                maxs[i] = np.where(rewards[i] > maxs[i - 1], rewards[i], maxs[i - 1])
                max_locations[:, i] = np.array(
                    (np.where(rewards[i] > maxs[i - 1], actions[:, i, 0], max_locations[:, i-1, 0]),
                     np.where(rewards[i] > maxs[i - 1], actions[:, i, 1], max_locations[:, i-1, 1]))).T
            states = tf.convert_to_tensor(actions, tf.float32)
            function_values = tf.convert_to_tensor(fcts(np.swapaxes(max_locations, 0, 1)), tf.float32)
            timestamp = time.strftime('%H:%M:%S', time.gmtime(time.time() - start))
            print("{} - {} minimization finished, {}"
                  .format(method_name, function_name, timestamp))
        else:
            raise NotImplementedError("{} algorithm hasn't been implemented yet".format(method_name))

        timestamp = time.strftime('%H:%M:%S', time.gmtime(time.time() - start))
        print("{} - {} plotting started, {}"
              .format(method_name, function_name, timestamp))
        performance, means, stds = plot(0,
                                        tf.float32,
                                        FUNCTIONS[function_name][1],
                                        plot_dir,
                                        n_start_pos,
                                        n_trans,
                                        N_rotations,
                                        N_input_noise,
                                        N_output_noise,
                                        domain,
                                        function_values=function_values,
                                        states=states,
                                        episode_length=episode_length,
                                        translations=translations,
                                        rotations=create_rotation_matrix(rotations)[0],
                                        input_noises=input_noises,
                                        output_noises=output_noises,
                                        name=function_name,
                                        train_episode_length=50,
                                        plot_trajectories=False,
                                        plot_all=False,
                                        log_summary=True)
        performances.append(performance)
        save_to_sql(means, stds, function_name, "", method_name, "", 0)
        mean_performance_over_time.append(tf.reduce_mean(means, axis=0).numpy())
        std_performance_over_time.append(tf.math.reduce_mean(stds, axis=0).numpy())
        timestamp = time.strftime('%H:%M:%S', time.gmtime(time.time() - start))
        print("{} - {} - final performance = {:.2f}, avg = {:.2f}, {}"
              .format(method_name, function_name, performance, np.mean(mean_performance_over_time[-1]), timestamp))

    mean_performance_over_time = np.array(mean_performance_over_time, np.float32)
    std_performance_over_time = np.array(std_performance_over_time, np.float32)
    mean_performance_over_time.tofile(file)
    std_performance_over_time.tofile(file2)


def plot_from_file(mean_file, std_file, output_length=min(200, episode_length)):
    mean_over_time = np.fromfile(mean_file, np.float32).reshape(len(FUNCTIONS), episode_length)
    std_over_time = np.fromfile(std_file, np.float32).reshape(len(FUNCTIONS), episode_length)
    plot_performance_over_time_with_stds(range(output_length),
                                         mean_over_time[:, :output_length],
                                         std_over_time[:, :output_length],
                                         FUNCTIONS.keys(),
                                         "{}-optimization".format(method_name),
                                         plot_dir,
                                         "performance over time.png",
                                         std_scale=0.25)

    plot_performance_by_function(FUNCTIONS.keys(), mean_over_time[:, 49], plot_dir, "final performances at 50 steps")
    plot_performance_by_function(FUNCTIONS.keys(), mean_over_time[:, output_length - 1], plot_dir,
                                 "final performances at {} steps"
                                 .format(output_length))

    summary = ["average performances by function at step {} \n".format(0)]
    for label, performance in zip(FUNCTIONS.keys(), mean_over_time[:, output_length - 1]):
        summary.append("{} function performance = {:.2f} \n".format(label, performance))

    summary.append("{} overall performance = {:.2f}".format(method_name, np.mean(mean_over_time[:, output_length - 1])))
    print(summary[-1])
    f = open(os.path.join(plot_dir, "summary.txt"), 'w')
    f.writelines(summary)


if __name__ == "__main__":
    start_time = time.time()
    arg_parser = argparse.ArgumentParser(description="name of function to plot")
    arg_parser.add_argument("-n",
                            "--name",
                            dest="name",
                            help="provide name of algorithm to evaluate",
                            type=str,
                            default="random_search")
    args = arg_parser.parse_args()

    method_name = args.name
    plot_dir = os.path.join(RUNS_DIR, method_name)
    if not os.path.isdir(plot_dir):
        os.makedirs(plot_dir)
    file_name = os.path.join(plot_dir, "performances.csv")
    file_name2 = os.path.join(plot_dir, "stds.csv")
    run(method_name, file_name, file_name2)
    plot_from_file(file_name, file_name2)
