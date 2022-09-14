import os

import gin
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

from tf_agents.utils import common

from agents.create_agent import create_agent
from definitons import RUNS_DIR, ROOT_DIR
from environments.create_environment import create_environment
from objective_functions.tf_objective_functions import FUNCTIONS
from training import train

ALGORITHM = "reinforce"
TRAINED_ON = "all"
# RUN_ID = "10_obs_0.9_gamma_100_episodes_1.0_trans_abs_env"
# RUN_ID = "10_obs_0.9_gamma_100_episodes_1.0_trans_abs_env"
# RUN_ID = "25_obs_1.0_gamma_1.0_trans_rel_env"
RUN_ID = "25_obs_giant_vnet_deep_pnet_rel_env"

FIG_NAME = "{}_top_perf_traj_{}".format(TRAINED_ON[:3], RUN_ID[-7:-4])

PLOT_DIR = os.path.join(ROOT_DIR, 'trajectories')

RUN_ON = [name for name in FUNCTIONS.keys()]
OPERATORS = ["control"]

EPISODE_LENGTH = 50
RUN_DIR = os.path.join(RUNS_DIR, ALGORITHM, TRAINED_ON, RUN_ID)
CHECKPOINT_DIR = os.path.join(RUN_DIR, 'checkpoint', 'policy')

CONFIG_FILE = os.path.join(RUN_DIR, 'config.gin')
N_START_POINTS = 1

BATCH_SIZE = len(RUN_ON) * len(OPERATORS)

COLORS = (
    'tab:orange', 'tab:red', 'purple', 'orchid', 'rosybrown', 'midnightblue',
    'orangered', 'tomato', 'maroon', 'gold', 'darkslategrey', 'lime', 'pink')

scatter_size = 4
LINEWIDTH = 1

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
            np.tile(np.array(((0.8, 0.8), (0.8, -0.8), (-0.8, 0.8), (-0.8, -0.8)), dtype=np.float32), (BATCH_SIZE, 1))
        )
    else:
        raise NotImplementedError("{} start points not implemented".format(N_START_POINTS))

    env = create_environment(environment_type,
                             RUN_ON,
                             [FUNCTIONS[name][0] for name in RUN_ON],
                             start_point,
                             EPISODE_LENGTH,
                             num_observations,
                             BATCH_SIZE * N_START_POINTS)

    agent = create_agent(agent_name,
                         env.observation_spec(),
                         env.action_spec(),
                         env.time_step_spec(),
                         global_step)

    policy_checkpointer = common.Checkpointer(
        ckpt_dir=CHECKPOINT_DIR,
        policy=agent.policy,
        global_step=global_step)

    load_status = policy_checkpointer.initialize_or_restore().expect_partial()
    load_status.assert_consumed()

    plot(collect_trajectories(env, agent.policy))


def plot(trajectories):
    domain = tf.constant([[-1, -1], [1, 1]], dtype=tf.float32)
    resolution = 5000
    X = tf.range(domain[0, 0], domain[1, 0], (domain[1, 0] - domain[0, 0]) / resolution)
    Y = tf.range(domain[0, 1], domain[1, 1], (domain[1, 1] - domain[0, 1]) / resolution)
    meshgrid = tf.convert_to_tensor(tf.meshgrid(X, Y))
    plt.subplots_adjust(left=0.05, bottom=0.05, right=0.95, top=0.95, wspace=0.1, hspace=0.1)
    fig, axs = plt.subplots(4, 2)
    fig.set_size_inches(6, 10)
    for i, ax in enumerate(axs.flatten()):
        image = FUNCTIONS[RUN_ON[i]][0](meshgrid)
        ax.set_title(RUN_ON[i].capitalize())
        p = plot_trajectory(ax, meshgrid, image, trajectories[i * N_START_POINTS: (i + 1) * N_START_POINTS], domain)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='5%', pad=0.1)
        fig.colorbar(p, cax=cax, orientation='vertical')

        if i % 2 == 1:
            ax.set_yticklabels([])
        else:
            ax.set_yticklabels([-1, None, 0, None, 1])
        if i > 5:
            ax.set_xticklabels([-1, None, 0, None, 1])
        else:
            ax.set_xticklabels([])
    plt.savefig(os.path.join(PLOT_DIR, FIG_NAME), dpi=400, transparent=True)
    plt.show()

def plot_trajectory(ax, meshgrid, image, states, domain):
    image = np.flipud(image)
    extent = (domain[0, 0], domain[1, 0], domain[0, 1], domain[1, 1])
    p = ax.imshow(image, cmap='viridis', extent=extent)
    image = np.flipud(image)
    levels = [-1, -0.75, -0.5, -0.25, 0.25, 0.5, 0.75, 0.85, 0.95, 0.999]
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
