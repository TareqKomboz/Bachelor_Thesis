import argparse
import os
import time

import numpy as np
import tensorflow as tf

from l2o.objective_functions.tf_objective_functions import FUNCTIONS
from l2o.definitions import RUNS_DIR
from l2o.evaluation.plot_utils import plot_performance_over_time_with_stds, plot, plot_performance_by_function

episode_length = 200


class NpFuncWrapper:
    def __init__(self, func, invert=False):
        self.func = func
        self.invert = invert
        self.max = -1.0
        self.trajectory = []

    def evaluate(self, X):
        X = tf.convert_to_tensor(X, tf.float32)
        reward = self.func(tf.transpose(X))
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


def plot_from_file(mean_file, std_file, output_length=min(200, episode_length)):
    mean_over_time = np.fromfile(mean_file, np.float32).reshape(len(FUNCTIONS), episode_length)
    std_over_time = np.fromfile(std_file, np.float32).reshape(len(FUNCTIONS), episode_length)
    plot_performance_over_time_with_stds(
        x=range(output_length),
        means=mean_over_time[:, :output_length],
        stds=std_over_time[:, :output_length],
        labels=FUNCTIONS.keys(),
        title="{}-optimization".format(method_name),
        plot_dir=plot_dir,
        filename="performance over time.png",
        std_scale=0.25
    )

    plot_performance_by_function(
        labels=FUNCTIONS.keys(),
        performances=mean_over_time[:, 49],
        plot_dir=plot_dir,
        name="final performances at 50 steps"
    )

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
