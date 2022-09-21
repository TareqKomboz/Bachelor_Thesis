import os

import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt


def plot_returns_and_losses(returns,
                            train_losses,
                            performances,
                            plot_dir,
                            agent_name,
                            function_name,
                            quick_eval_interval):
    returns_file = os.path.join(plot_dir, "train_returns.csv")
    if os.path.isfile(returns_file):
        old_returns = np.fromfile(returns_file, dtype=np.float32)
        returns = np.concatenate([old_returns, returns], axis=0)
    returns.tofile(returns_file)

    losses_file = os.path.join(plot_dir, "train_losses.csv")
    if os.path.isfile(losses_file):
        old_losses = np.fromfile(losses_file, dtype=np.float32)
        train_losses = np.concatenate([old_losses, train_losses], axis=0)
    train_losses.tofile(losses_file)

    performance_file = os.path.join(plot_dir, "train_performance.csv")
    if os.path.isfile(performance_file):
        old_performances = np.fromfile(performance_file, dtype=np.float32)
        old_performances = old_performances.reshape((int(len(old_performances) / 2), 2))
        performances = np.concatenate([old_performances, performances], axis=0)
    performances.flatten().tofile(performance_file)

    xx = np.arange(0, len(returns))
    plt.plot(xx, returns)
    plt.ylim(0, 1)
    plt.ylabel("average return")
    plt.xlabel("train step")
    plt.title("{} - {} returns".format(agent_name, function_name))
    plt.grid()
    plt.savefig(os.path.join(plot_dir, "train_returns"), transparent=True)
    plt.clf()
    xx = np.arange(0, len(train_losses))
    plt.plot(xx, train_losses)
    plt.ylabel("train losses")
    plt.xlabel("train step")
    plt.grid()
    plt.title("{} - {} train losses".format(agent_name, function_name))
    plt.savefig(os.path.join(plot_dir, "train_losses"), transparent=True)
    plt.clf()
    xx = np.arange(0, (len(performances) * quick_eval_interval), quick_eval_interval)
    plt.plot(xx, performances[:, 0])
    plt.plot(xx, performances[:, 1])
    plt.legend(["final", "average"])
    plt.ylabel("performances")
    plt.xlabel("train step")
    plt.grid()
    plt.ylim(0, 1)
    plt.title("{} - {} performances".format(agent_name, function_name))
    plt.savefig(os.path.join(plot_dir, "performances"), transparent=True)
    plt.clf()
    plt.plot(xx, performances[:, 0])
    plt.ylabel("final performances")
    plt.xlabel("train step")
    plt.ylim(0, 1)
    plt.grid()
    plt.title("{} - {} final performances".format(agent_name, function_name))
    plt.savefig(os.path.join(plot_dir, "final_performances"), transparent=True)
    plt.clf()


def build_eval_params(n_start_pos, input_dimension):
    N_start_pos = (n_start_pos + 1) ** input_dimension

    my_list = []
    for i in range(input_dimension):
        my_list.append(tf.range(-0.9, 0.91, 1.8 / n_start_pos))
    start_mesh = tf.reshape(tf.transpose(tf.meshgrid(*tuple(my_list))), (N_start_pos, input_dimension))

    # Starting positions
    starting_positions = start_mesh

    return starting_positions


def split_function_values_and_states(function_values, states):
    f = function_values
    split_function_values = [f]
    f = states
    split_states = [f]

    return split_function_values, split_states
