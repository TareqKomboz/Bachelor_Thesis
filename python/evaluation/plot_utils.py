import logging
import os

import numpy as np
import matplotlib.colors as mcolors
from matplotlib import pyplot as plt
import tensorflow as tf


CMAP = 'viridis'
ALPHA = 0.2
COLORS_low_contrast = (
    'tab:orange', 'tab:red', 'darkolivegreen', 'indigo', 'darkorange', 'purple', 'rosybrown', 'midnightblue',
    'orchid','orangered', 'tomato', 'maroon', 'gold', 'darkslategrey', 'lime', 'pink'
)


def plot_performance_by_function(labels, performances, plot_dir, name):
    width = 0.2
    plt.bar(labels, performances, width, color=mcolors.TABLEAU_COLORS)
    plt.xticks(rotation=30)
    plt.ylim(0, 1)
    plt.yticks(np.arange(0.0, 1.1, 0.1))
    plt.subplots_adjust(top=0.85, bottom=0.15, right=0.95, left=0.05)
    plt.title(name)
    plt.grid(axis='y')
    plt.savefig(os.path.join(plot_dir, name), transparent=True)
    plt.clf()
    plt.subplots_adjust(top=0.9, bottom=0.1, right=0.9, left=0.125)


def plot_performance_over_time_with_stds(x, means, stds, labels, title, plot_dir, filename, std_scale=0.1):
    plt.ylim(0, 1)
    # plt.tight_layout()
    line = []
    colors = [color for color in mcolors.TABLEAU_COLORS.values()]
    for i, (mean, std) in enumerate(zip(means, stds)):
        line.append(plt.plot(x, mean, color=colors[i]))
    plt.legend(labels, loc=4)
    for i, (mean, std) in enumerate(zip(means, stds)):
        plt.fill_between(x, mean, mean - std_scale * std, color=colors[i], alpha=0.2)
        plt.fill_between(x, mean, mean + std_scale * std, color=colors[i], alpha=0.2)
    plt.grid()
    plt.yticks(np.arange(0, 1.1, 0.1))
    plt.xlabel("steps")
    plt.ylabel("performance")
    plt.title(title)
    plt.savefig(os.path.join(plot_dir, filename), transparent=True)
    plt.clf()


def plot(step_counter, plot_dir, function_values, name):
    plot_dir = os.path.join(plot_dir, name)

    # control
    if not os.path.isdir(plot_dir):
        os.makedirs(plot_dir)

    average_reward_over_batches_and_actions = tf.reduce_mean(function_values)
    reward_stds_over_batches_and_actions = tf.convert_to_tensor([tf.math.reduce_std(function_values)])

    average_return_over_batch = tf.reduce_mean(tf.reduce_sum(input_tensor=function_values, axis=1, keepdims=True))
    return_stds_over_batch = tf.convert_to_tensor([tf.math.reduce_std(tf.reduce_sum(
        input_tensor=function_values,
        axis=1,
        keepdims=True
    ))])

    average_final_objective_function_value_over_batch = tf.reduce_mean(function_values[:, -1])
    final_objective_function_value_stds_over_batch = tf.convert_to_tensor([tf.math.reduce_std(function_values[:, -1])])

    average_max_reward_of_episode_over_batches = tf.reduce_mean(tf.reduce_max(function_values, axis=1))
    max_reward_of_episode_stds_over_batches = tf.convert_to_tensor([tf.math.reduce_std(tf.reduce_max(function_values, axis=1))])

    reward_means_over_batch = tf.convert_to_tensor([tf.reduce_mean(function_values, axis=0)])
    reward_stds_over_batch = tf.convert_to_tensor([tf.math.reduce_std(function_values, axis=0)])

    # summary = [
    #     "{} evaluation results at step {} \n".format(name, step_counter),
    #     "average reward={:.2f}, average final reward={:.2f}, average max reward={:.2f} \n".format(
    #         average_reward_over_batches_and_actions,
    #         reward_stds_over_batches_and_actions,
    #         average_return_over_batch,
    #         return_stds_over_batch,
    #         average_final_objective_function_value_over_batch,
    #         final_objective_function_value_stds_over_batch,
    #         average_max_reward_of_episode_over_batches,
    #         max_reward_of_episode_stds_over_batches
    #     )
    # ]

    summary = [
        "{} evaluation results at step {} \n".format(name, step_counter),
        "average_reward_over_batches_and_actions={:.2f}, reward_stds_over_batches_and_actions={} \n".format(
            average_reward_over_batches_and_actions,
            reward_stds_over_batches_and_actions.numpy()
        ),
        "average_return_over_batch={:.2f}, return_stds_over_batch={} \n".format(
            average_return_over_batch,
            return_stds_over_batch.numpy()
        ),
        "average_final_objective_function_value_over_batch={:.2f}, final_objective_function_value_stds_over_batch={} \n".format(
            average_final_objective_function_value_over_batch,
            final_objective_function_value_stds_over_batch.numpy()
        ),
        "average_max_reward_of_episode_over_batches={:.2f}, max_reward_of_episode_stds_over_batches={} \n".format(
            average_max_reward_of_episode_over_batches,
            max_reward_of_episode_stds_over_batches.numpy()
        )
    ]
    f = open(os.path.join(plot_dir, "summary.txt"), 'w')
    f.writelines(summary)

    # plot x=episode_length, y=average_reward
    plot_performance_over_time_with_stds(
        x=range(len(reward_means_over_batch[0])),
        means=reward_means_over_batch,
        stds=reward_stds_over_batch,
        labels=["control"],
        title="Convergence by category",
        plot_dir=plot_dir,
        filename="performance over time",
        std_scale=0.25
    )

    return average_final_objective_function_value_over_batch, reward_means_over_batch, reward_stds_over_batch
