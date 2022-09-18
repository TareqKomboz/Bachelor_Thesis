import logging
import os

import numpy as np
import matplotlib.colors as mcolors
from matplotlib import pyplot as plt
import tensorflow as tf

from evaluation.evaluation_utils import split_function_values_and_states

CMAP = 'viridis'
ALPHA = 0.2
COLORS_low_contrast = (
    'tab:orange', 'tab:red', 'darkolivegreen', 'indigo', 'darkorange', 'purple', 'rosybrown', 'midnightblue', 'orchid',
    'orangered', 'tomato', 'maroon', 'gold', 'darkslategrey', 'lime', 'pink')

COLORS_high_contrast = ()


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


def plot_performance_over_time(x, performances, labels, title, plot_dir):
    plt.ylim(0, 1)
    plt.plot(x, performances)
    plt.legend(labels)
    plt.grid()
    plt.yticks(np.arange(0, 1.1, 0.1))
    plt.xlabel("steps")
    plt.ylabel("performance")
    plt.title(title)
    plt.savefig(os.path.join(plot_dir, "performance_over_time"), transparent=True)
    plt.clf()


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


def plot(step_counter,
         plot_dir,
         n_start_pos,
         function_values,
         states,
         name,
         train_episode_length,
         log_summary=False
         ):
    plot_dir = os.path.join(plot_dir, name)

    n_start_pos = (n_start_pos + 1) ** 2

    # control
    if not os.path.isdir(plot_dir):
        os.makedirs(plot_dir)

    control_rewards = function_values[:n_start_pos]

    avg_control_reward = tf.reduce_mean(control_rewards)
    rewards = [avg_control_reward]

    avg_control_final = tf.reduce_mean(control_rewards[:, -1])
    finals = [avg_control_final]

    avg_control_train_final = tf.reduce_mean(control_rewards[:, train_episode_length - 1])
    train_finals = [avg_control_train_final]

    avg_control_max = tf.reduce_mean(tf.reduce_max(control_rewards, axis=1))
    maxs = [avg_control_max]

    overall_avg_performance = tf.reduce_mean(tf.convert_to_tensor(rewards))
    overall_final_performance = tf.reduce_mean(tf.convert_to_tensor(finals))
    overall_train_final_performance = tf.reduce_mean(tf.convert_to_tensor(train_finals))
    overall_max_performance = tf.reduce_mean(tf.convert_to_tensor(maxs))

    summary = [
        "{} evaluation results at step {} \n".format(name, step_counter),
        "overall return={:.2f}, train_final={:.2f}, final={:.2f}, max={:.2f} \n".format(
            overall_avg_performance, overall_train_final_performance, overall_final_performance, overall_max_performance
        ), "control return={:.2f}, train_final={:.2f}, final={:.2f}, max={:.2f} \n".format(
            avg_control_reward, avg_control_train_final, avg_control_final, avg_control_max
        )
    ]
    if log_summary:
        for line in summary:
            logging.info(line.strip("\n"))
    f = open(os.path.join(plot_dir, "summary.txt"), 'w')
    f.writelines(summary)

    function_values, states = split_function_values_and_states(function_values, states)

    means = tf.convert_to_tensor([tf.reduce_mean(category, axis=0) for category in function_values])
    stds = tf.convert_to_tensor([tf.math.reduce_std(category, axis=0) for category in function_values])

    plot_performance_over_time_with_stds(
        range(len(means[0])),
        means,
        stds,
        ["control"],
        "convergence by category",
        plot_dir,
        "performance over time",
        std_scale=0.25
    )

    plot_performance_over_time_with_stds(
        range(train_episode_length),
        means[:, :train_episode_length],
        stds[:, :train_episode_length],
        ["control"],
        "convergence by category",
        plot_dir,
        "performance over time {} steps".format(train_episode_length),
        std_scale=0.25
    )

    return overall_final_performance, means, stds
