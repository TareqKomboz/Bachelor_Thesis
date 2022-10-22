import os

import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt


def plot_returns_and_losses(
        train_rewards,
        train_losses,
        evaluation_performances,
        plot_dir,
        input_dimension,
        number_free_parameters,
        function_name,
        quick_evaluation_interval):

    xx = np.arange(0, len(train_rewards))
    plt.plot(xx, train_rewards)
    plt.xlim(0, len(train_rewards))
    plt.ylim(0, 1)
    plt.ylabel("Training Reward")
    plt.xlabel("Training Iteration Step")
    plt.title("{}D {} - {} Free - Rraining Rewards".format(input_dimension, function_name, number_free_parameters))
    plt.grid()
    plt.savefig(os.path.join(plot_dir, "train_rewards"), transparent=True)
    plt.clf()

    xx = np.arange(0, len(train_losses))
    plt.xlim(0, len(train_rewards))
    plt.ylim(0)
    plt.plot(xx, train_losses)
    plt.ylabel("Training Loss")
    plt.xlabel("Training Iteration Step")
    plt.grid()
    plt.title("{}D {} - {} Free - Train Loss".format(input_dimension, function_name, number_free_parameters))
    plt.savefig(os.path.join(plot_dir, "train_losses"), transparent=True)
    plt.clf()

    xx = np.arange(0, (len(evaluation_performances) * quick_evaluation_interval), quick_evaluation_interval)
    plt.plot(xx, evaluation_performances[:, 0])
    plt.plot(xx, evaluation_performances[:, 1])
    plt.legend(["Average Final Reward per Episode", "Average Reward per Action"])
    plt.ylabel("Evaluation Performances")
    plt.xlabel("Training Iteration Step")
    plt.grid()
    plt.ylim(0, 1)
    plt.title("{}D {} - {} Free - Evaluation Performance".format(input_dimension, function_name, number_free_parameters))
    plt.savefig(os.path.join(plot_dir, "evaluation_performances"), transparent=True)
    plt.clf()


def build_evaluation_parameters(n_start_pos, input_dimension):
    N_start_pos = n_start_pos ** input_dimension

    my_list = []
    my_range = tf.range(-0.9, 0.91, 1.8 / (n_start_pos - 1))
    for i in range(input_dimension):
        my_list.append(my_range)

    mesh = tf.meshgrid(*tuple(my_list))
    transposed_mesh = tf.transpose(mesh)
    reshaped_mesh = tf.reshape(transposed_mesh, (N_start_pos, input_dimension))

    return reshaped_mesh


def build_evaluation_parameters_new(n_start_pos, input_dimension):
    N_start_pos = n_start_pos ** input_dimension

    my_list = []
    my_linspace = np.linspace(-0.9, 0.9, num=n_start_pos, dtype=np.float32)
    for i in range(input_dimension):
        my_list.append(my_linspace)

    mesh = np.meshgrid(*tuple(my_list), sparse=True)
    transposed_mesh = np.transpose(mesh)
    reshaped_mesh = np.reshape(transposed_mesh, (1, input_dimension))

    return reshaped_mesh
