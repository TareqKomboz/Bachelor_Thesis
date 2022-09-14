import math
import os

import gin
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

from tf_agents.utils import common

from agents.create_agent import create_agent
from common.utils import create_rotation_matrix
from definitons import RUNS_DIR, ROOT_DIR
from environments.create_environment import create_environment
from objective_functions.tf_objective_functions import FUNCTIONS
from training import train

ALGORITHM = "reinforce"
TRAINED_ON = "cross-in-tray"
# RUN_ID = "10_obs_0.9_gamma_1.0_trans_abs_env"
# RUN_ID = "10_obs_0.9_gamma_100_episodes_1.0_trans_abs_env"
RUN_ID = "25_obs_1.0_gamma_1.0_trans_rel_env"
# RUN_ID = "25_obs_double_pnet_rel_env"
# RUN_ID = "25_obs_giant_vnet_deep_pnet_rel_env"
RUN_ON = "rosenbrock"

include_operators = (True, False, False, False, False)

FIG_NAME = "{}_{}_run_on_{}".format(TRAINED_ON[:3], RUN_ID[-7:-4], RUN_ON[:3])

PLOT_DIR = os.path.join(ROOT_DIR, 'trajectories')
translation = (-0.8, -0.8)
rotation = math.pi * 0.7
output_noise = 0.015
input_noise = 0.015

N_START_POINTS = 1
EPISODE_LENGTH = 200

RUN_DIR = os.path.join(RUNS_DIR, ALGORITHM, TRAINED_ON, RUN_ID)
CHECKPOINT_DIR = os.path.join(RUN_DIR, 'checkpoint', 'policy')

CONFIG_FILE = os.path.join(RUN_DIR, 'config.gin')

BATCH_SIZE = np.count_nonzero(include_operators)

COLORS = (
    'tab:orange', 'tab:red', 'purple', 'orchid', 'rosybrown', 'midnightblue',
    'orangered', 'tomato', 'maroon', 'gold', 'darkslategrey', 'lime', 'pink')

scatter_size = 4
LINEWIDTH = 1

OPERATORS = ["Control", "Translation", "Rotation", "Input noise", "Output noise"]
OPERATORS = [op for (op, inc) in zip(OPERATORS, include_operators) if inc]
RESOLUTION = 640
DOMAIN = tf.convert_to_tensor([[-1, -1], [1, 1]], dtype=tf.float32)
X = tf.range(DOMAIN[0][0], DOMAIN[1][0], (DOMAIN[1][0] - DOMAIN[0][0]) / RESOLUTION)
Y = tf.range(DOMAIN[0][1], DOMAIN[1][1], (DOMAIN[1][1] - DOMAIN[0][1]) / RESOLUTION)
MESHGRID = tf.convert_to_tensor(tf.meshgrid(X, Y))

FUNCTION = FUNCTIONS[RUN_ON][0]


@gin.configurable
def main(environment_type,
         agent_name,
         function_names,
         num_observations):
    if not os.path.isdir(PLOT_DIR):
        os.mkdir(PLOT_DIR)

    if not os.path.isdir(CHECKPOINT_DIR):
        raise FileNotFoundError("No Checkpoints available at {}, evaluation is aborting".format(CHECKPOINT_DIR))
    global_step = tf.compat.v1.train.get_or_create_global_step()

    if N_START_POINTS == 1:
        start_point = tf.constant(0.8, shape=(BATCH_SIZE, 2), dtype=tf.float32)
    elif N_START_POINTS == 4:
        start_point = tf.convert_to_tensor(
            np.array(((0.8, 0.8), (0.8, -0.8), (-0.8, 0.8), (-0.8, -0.8)), dtype=np.float32)
        )
    else:
        raise NotImplementedError("{} start points not implemented".format(N_START_POINTS))
    envs = []
    function = (FUNCTION,)
    if include_operators[0]:
        envs.append(create_environment(environment_type,
                                       RUN_ON,
                                       function,
                                       start_point,
                                       EPISODE_LENGTH,
                                       num_observations,
                                       N_START_POINTS))
    if include_operators[1]:
        envs.append(create_environment(environment_type,
                                       RUN_ON,
                                       function,
                                       start_point,
                                       EPISODE_LENGTH,
                                       num_observations,
                                       N_START_POINTS,
                                       translation=tf.tile((translation,), (N_START_POINTS, 1))))
    if include_operators[2]:
        envs.append(create_environment(environment_type,
                                       RUN_ON,
                                       function,
                                       start_point,
                                       EPISODE_LENGTH,
                                       num_observations,
                                       N_START_POINTS,
                                       rotation=rotation))
    if include_operators[3]:
        envs.append(create_environment(environment_type,
                                       RUN_ON,
                                       function,
                                       start_point,
                                       EPISODE_LENGTH,
                                       num_observations,
                                       N_START_POINTS,
                                       output_noise=output_noise))
    if include_operators[4]:
        envs.append(create_environment(environment_type,
                                       RUN_ON,
                                       function,
                                       start_point,
                                       EPISODE_LENGTH,
                                       num_observations,
                                       N_START_POINTS,
                                       input_noise=input_noise))

    agent = create_agent(agent_name,
                         envs[0].observation_spec(),
                         envs[0].action_spec(),
                         envs[0].time_step_spec(),
                         global_step)

    policy_checkpointer = common.Checkpointer(
        ckpt_dir=CHECKPOINT_DIR,
        policy=agent.policy,
        global_step=global_step)

    load_status = policy_checkpointer.initialize_or_restore().expect_partial()
    load_status.assert_consumed()
    states = []
    for env in envs:
        states.append(collect_trajectories(env, agent.policy))
    plot(tf.concat(states, axis=0))


def draw_images():
    images = [FUNCTION(MESHGRID)]
    translated_mesh = tf.convert_to_tensor((
        tf.subtract(MESHGRID[0], -translation[0]),
        tf.subtract(MESHGRID[1], -translation[1])
    ))
    images.append(FUNCTION(translated_mesh))
    print("translation", end="")
    R = create_rotation_matrix(rotation)
    rotated_mesh = tf.convert_to_tensor((
        MESHGRID[0] * R[0][0] +
        MESHGRID[1] * R[0][1],
        MESHGRID[0] * R[1][0] +
        MESHGRID[1] * R[1][1]
    ))
    images.append(FUNCTION(rotated_mesh))
    print(", rotation", end="")
    in_noisy_mesh = MESHGRID + tf.random.normal(MESHGRID.shape, mean=0.0, stddev=input_noise)
    images.append(FUNCTION(in_noisy_mesh))
    print(", input_noise", end="")
    image = FUNCTION(MESHGRID)
    image += tf.random.normal(image.shape, mean=0.0, stddev=output_noise)
    images.append(image)
    print(", output_noise drawn")
    return images


def plot(trajectories):
    domain = tf.constant([[-1, -1], [1, 1]], dtype=tf.float32)
    plt.subplots_adjust(left=0.05, bottom=0.05, right=0.85, top=0.95, wspace=0.1, hspace=0.1)
    fig, axs = plt.subplots(1, BATCH_SIZE)
    fig.set_size_inches(5, 3)
    images = draw_images()
    images = [image for (bool, image) in zip(include_operators, images) if bool]
    p = []
    if not hasattr(axs, '__iter__'):
        axs = [axs,]
    for i, (ax, image) in enumerate(zip(axs, images)):
        #ax.set_title(OPERATORS[i])
        p.append(
            plot_trajectory(ax, MESHGRID, image, trajectories[i * N_START_POINTS: (i + 1) * N_START_POINTS], domain))

        if i > 0:
            ax.set_yticklabels([])
        else:
            ax.set_yticklabels([-1, None, 0, None, 1])

        plt.locator_params(nbins=5)
        ax.set_xticklabels([-1, None, 0, None, 1])

    cax = plt.gcf().add_axes([0.78, 0.11, 0.02, 0.77])
    plt.colorbar(p[0], cax=cax, orientation='vertical')
    plt.savefig(os.path.join(PLOT_DIR, FIG_NAME), dpi=400, transparent=True)
    plt.show()


def plot_trajectory(ax, meshgrid, image, states, domain):
    image = np.flipud(image)
    extent = (domain[0, 0], domain[1, 0], domain[0, 1], domain[1, 1])
    p = ax.imshow(image, cmap='viridis', extent=extent)
    image = np.flipud(image)
    levels = [-1, -0.75, -0.5, -0.25, 0.2, 0.5, 0.75, 0.85, 0.95, 0.999]
    ax.contour(meshgrid[0], meshgrid[1], image, levels, colors='k', alpha=1.0, extent=extent, linewidths=0.5)
    ax.contour(meshgrid[0], meshgrid[1], image, [0, ], colors='k', extent=extent, linewidths=0.75, linestyles='solid')
    x, y = states[0, :, 0], states[0, :, 1]
    ax.plot(x, y, color=COLORS[0], alpha=1.0, linewidth=LINEWIDTH)
    ax.scatter(x[1:-1], y[1:-1], color=COLORS[0], s=scatter_size, alpha=1.0)
    ax.scatter(x[-1], y[-1], color='tab:red', alpha=1.0, marker='s', s=scatter_size)
    for i in range(1, N_START_POINTS):
        x, y = states[i, :, 0], states[i, :, 1]
        ax.plot(x, y, color=COLORS[i], alpha=1.0, linewidth=LINEWIDTH)
        ax.scatter(x[1:-1], y[1:-1], color=COLORS[i], s=scatter_size, alpha=1.0)
        ax.scatter(x[-1], y[-1], color='tab:red', alpha=1.0, marker='s', s=scatter_size)
    return p


def collect_trajectories(env, policy):
    time_step = env.reset()
    for i in range(EPISODE_LENGTH - 1):
        policy_step = policy.action(time_step)
        time_step = env.step(policy_step.action)
    return env.get_states()


if __name__ == "__main__":
    gin.parse_config_file(CONFIG_FILE)
    main()
