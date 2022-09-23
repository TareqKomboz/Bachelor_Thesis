import os

import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt


def plot_returns_and_losses(
        train_rewards,
        train_losses,
        evaluation_performances,
        plot_dir,
        agent_name,
        function_name,
        quick_evaluation_interval):
    # todo: generate file/array
    # returns_file = os.path.join(plot_dir, "train_returns.csv")
    # if os.path.isfile(returns_file):
    #     old_returns = np.fromfile(returns_file, dtype=np.float32)
    #     returns = np.concatenate([old_returns, returns], axis=0)
    # returns.tofile(returns_file)
    #
    # losses_file = os.path.join(plot_dir, "train_losses.csv")
    # if os.path.isfile(losses_file):
    #     old_losses = np.fromfile(losses_file, dtype=np.float32)
    #     train_losses = np.concatenate([old_losses, train_losses], axis=0)
    # train_losses.tofile(losses_file)
    #
    # performance_file = os.path.join(plot_dir, "train_performance.csv")
    # if os.path.isfile(performance_file):
    #     old_performances = np.fromfile(performance_file, dtype=np.float32)
    #     old_performances = old_performances.reshape((int(len(old_performances) / 2), 2))
    #     performances = np.concatenate([old_performances, performances], axis=0)
    # performances.flatten().tofile(performance_file)

    xx = np.arange(0, len(train_rewards))
    plt.plot(xx, train_rewards)
    plt.ylim(0, 1)
    plt.ylabel("average reward per action")
    plt.xlabel("train step")
    plt.title("{} - {} returns".format(agent_name, function_name))
    plt.grid()
    plt.savefig(os.path.join(plot_dir, "train_rewards"), transparent=True)
    plt.clf()

    xx = np.arange(0, len(train_losses))
    plt.plot(xx, train_losses)
    plt.ylabel("train losse")
    plt.xlabel("train step")
    plt.grid()
    plt.title("{} - {} train losses".format(agent_name, function_name))
    plt.savefig(os.path.join(plot_dir, "train_losses"), transparent=True)
    plt.clf()

    xx = np.arange(0, (len(evaluation_performances) * quick_evaluation_interval), quick_evaluation_interval)
    plt.plot(xx, evaluation_performances[:, 0])
    plt.plot(xx, evaluation_performances[:, 1])
    plt.legend(["average final reward per episode", "average reward per action"])
    plt.ylabel("evaluation performances")
    plt.xlabel("train step")
    plt.grid()
    plt.ylim(0, 1)
    plt.title("{} - {} evaluation performances".format(agent_name, function_name))
    plt.savefig(os.path.join(plot_dir, "evaluation_performances"), transparent=True)
    plt.clf()

    plt.plot(xx, evaluation_performances[:, 0])
    plt.ylabel("final evaluation_performances")
    plt.xlabel("train step")
    plt.ylim(0, 1)
    plt.grid()
    plt.title("{} - {} final evaluation_performances".format(agent_name, function_name))
    plt.savefig(os.path.join(plot_dir, "final_evaluation_performances"), transparent=True)
    plt.clf()


def build_evaluation_params(n_start_pos, input_dimension):
    N_start_pos = (n_start_pos + 1) ** input_dimension

    my_list = []
    for i in range(input_dimension):
        my_list.append(tf.range(-0.9, 0.91, 1.8 / n_start_pos))
    start_mesh = tf.reshape(tf.transpose(tf.meshgrid(*tuple(my_list))), (N_start_pos, input_dimension))

    # Starting positions
    starting_positions = start_mesh

    return starting_positions
