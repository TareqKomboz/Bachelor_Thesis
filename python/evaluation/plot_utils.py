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


def plot_trajectory(meshgrid, rewards, states, image, plot_dir, name, domain, N_start_pos, plot_all):
    image = np.flipud(image)
    # image = np.fliplr(image)
    extent = (domain[0, 0], domain[1, 0], domain[0, 1], domain[1, 1])
    plt.imshow(image, cmap=CMAP, extent=extent)
    image = np.flipud(image)
    levels = [-1, -0.75, -0.5, -0.25, 0.25, 0.5, 0.75, 0.85, 0.95, 0.999]
    plt.contour(meshgrid[0], meshgrid[1], image, levels, colors='k', alpha=1.0, extent=extent, linewidths=0.5)
    plt.contour(meshgrid[0], meshgrid[1], image, [0, ], colors='k', extent=extent, linewidths=0.75, linestyles='solid')
    colors = COLORS_low_contrast
    x, y = states[0, :, 0], states[0, :, 1]
    plt.plot(x, y, color=colors[0], alpha=1.0)
    plt.scatter(x[1:-1], y[1:-1], color=colors[0], alpha=1.0)
    plt.scatter(x[-1], y[-1], color='tab:red', alpha=1.0, marker='s')
    plt.xticks([], labels=[])
    plt.yticks([], labels=[])
    avg_reward = tf.reduce_mean(rewards)
    final_reward = tf.reduce_mean(rewards[:, -1])
    max_reward = tf.reduce_mean(tf.reduce_max(rewards, axis=1))
    if plot_all:
        for i in range(1, N_start_pos):
            x, y = states[i, :, 0], states[i, :, 1]
            plt.plot(x, y, color=colors[i], alpha=ALPHA)

        plt.title("final: {:.2f}, avg: {:.2f}, max: {:.2f}".format(final_reward, avg_reward, max_reward))

    else:
        avg_reward_example = tf.reduce_mean(rewards[0])
        final_reward_example = rewards[0, -1]
        max_reward_example = tf.reduce_max(rewards[0])
        plt.title("final: {:.2f} ({:.2f}), avg: {:.2f} ({:.2f}), max: {:.2f} ({:.2f})"
                  .format(final_reward_example, final_reward,
                          avg_reward_example, avg_reward,
                          max_reward_example, max_reward))

    plt.savefig(os.path.join(plot_dir, "{}.png".format(name)), transparent=True)
    plt.clf()


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
         dtype,
         plotting_function,
         plot_dir,
         n_start_pos,
         domain,
         function_values,
         states,
         episode_length,
         name,
         train_episode_length,
         plot_trajectories=False,
         plot_all=False,
         log_summary=False
         ):
    plot_dir = os.path.join(plot_dir, name)

    N_start_pos = (n_start_pos + 1) ** 2

    X = tf.range(domain[0, 0], domain[1, 0], (domain[1, 0] - domain[0, 0]) / 640)
    Y = tf.range(domain[0, 1], domain[1, 1], (domain[1, 1] - domain[0, 1]) / 640)
    meshgrid = tf.convert_to_tensor(tf.meshgrid(X, Y))

    # control
    if not os.path.isdir(plot_dir):
        os.makedirs(plot_dir)

    control_rewards = function_values[:N_start_pos]
    control_states = states[:N_start_pos]
    image = plotting_function(meshgrid)
    plot_trajectory(meshgrid, control_rewards, control_states, image, plot_dir,
                    "control-trajectory", domain, N_start_pos, plot_all)


    avg_control_reward = tf.reduce_mean(control_rewards)
    rewards = [avg_control_reward]

    avg_control_final = tf.reduce_mean(control_rewards[:, -1])
    finals = [avg_control_final]

    avg_control_train_final = tf.reduce_mean(control_rewards[:, train_episode_length - 1])
    train_finals = [avg_control_train_final]

    avg_control_max = tf.reduce_mean(tf.reduce_max(control_rewards, axis=1))
    maxs = [avg_control_max]

    labels = ['control']
    x = np.arange(len(labels))
    width = 0.2

    plt.bar(x, rewards, width)
    plt.xticks(x, labels)
    plt.title("return")
    plt.ylim(0, 1)
    plt.yticks(np.arange(0, 1.1, 0.1))
    plt.grid(axis='y')
    plt.savefig(os.path.join(plot_dir, "summary-return"), transparent=True)
    plt.clf()

    plt.bar(x, finals, width)
    plt.xticks(x, labels)
    plt.ylim(0, 1)
    plt.yticks(np.arange(0, 1.1, 0.1))
    plt.grid(axis='y')
    plt.title("Average final reward")
    plt.savefig(os.path.join(plot_dir, "summary-final"), transparent=True)
    plt.clf()

    plt.bar(x, train_finals, width)
    plt.xticks(x, labels)
    plt.ylim(0, 1)
    plt.yticks(np.arange(0, 1.1, 0.1))
    plt.grid(axis='y')
    plt.title("Average reward at {} steps".format(train_episode_length))
    plt.savefig(os.path.join(plot_dir, "summary-train-final"), transparent=True)
    plt.clf()

    plt.bar(x, maxs, width)
    plt.xticks(x, labels)
    plt.ylim(0, 1)
    plt.yticks(np.arange(0, 1.1, 0.1))
    plt.grid(axis='y')
    plt.title("Average max reward")
    plt.savefig(os.path.join(plot_dir, "summary-max"), transparent=True)
    plt.clf()

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

    plot_performance_over_time_with_stds(range(len(means[0])),
                                         means,
                                         stds,
                                         ["control", "translation", "rotation", "input noise", "output_noise"],
                                         "convergence by category",
                                         plot_dir,
                                         "performance over time",
                                         std_scale=0.25)

    plot_performance_over_time_with_stds(range(train_episode_length),
                                         means[:, :train_episode_length],
                                         stds[:, :train_episode_length],
                                         ["control", "translation", "rotation", "input noise", "output_noise"],
                                         "convergence by category",
                                         plot_dir,
                                         "performance over time {} steps".format(train_episode_length),
                                         std_scale=0.25)

    return overall_final_performance, means, stds


