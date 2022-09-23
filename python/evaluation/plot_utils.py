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


def plot(input_dimension, step_counter, plot_dir, n_start_pos, function_values, name, episode_length, log_summary=False):
    plot_dir = os.path.join(plot_dir, name)

    N_start_pos = (n_start_pos + 1) ** input_dimension

    # control
    if not os.path.isdir(plot_dir):
        os.makedirs(plot_dir)

    control_rewards = function_values[:N_start_pos]

    average_control_reward = tf.reduce_mean(control_rewards)
    rewards = [average_control_reward]

    average_control_final = tf.reduce_mean(control_rewards[:, -1])
    finals = [average_control_final]

    average_control_train_final = tf.reduce_mean(control_rewards[:, episode_length - 1])
    train_finals = [average_control_train_final]

    average_control_max = tf.reduce_mean(tf.reduce_max(control_rewards, axis=1))
    maxs = [average_control_max]

    overall_average_performance = tf.reduce_mean(tf.convert_to_tensor(rewards))
    overall_final_performance = tf.reduce_mean(tf.convert_to_tensor(finals))
    overall_train_final_performance = tf.reduce_mean(tf.convert_to_tensor(train_finals))
    overall_max_performance = tf.reduce_mean(tf.convert_to_tensor(maxs))

    summary = [
        "{} evaluation results at step {} \n".format(name, step_counter),
        "overall return={:.2f}, train_final={:.2f}, final={:.2f}, max={:.2f} \n".format(
            overall_average_performance,
            overall_train_final_performance,
            overall_final_performance,
            overall_max_performance
        ),
        "control return={:.2f}, train_final={:.2f}, final={:.2f}, max={:.2f} \n".format(
            average_control_reward,
            average_control_train_final,
            average_control_final,
            average_control_max
        )
    ]
    if log_summary:
        for line in summary:
            logging.info(line.strip("\n"))
    f = open(os.path.join(plot_dir, "summary.txt"), 'w')
    f.writelines(summary)

    means = tf.convert_to_tensor([tf.reduce_mean(function_values, axis=0)])
    stds = tf.convert_to_tensor([tf.math.reduce_std(function_values, axis=0)])

    plot_performance_over_time_with_stds(
        x=range(len(means[0])),
        means=means,
        stds=stds,
        labels=["control"],
        title="convergence by category",
        plot_dir=plot_dir,
        filename="performance over time",
        std_scale=0.25
    )

    return overall_final_performance, means, stds
