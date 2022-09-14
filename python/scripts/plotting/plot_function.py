import argparse
import os
import time

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

from definitons import ROOT_DIR
from objective_functions.tf_objective_functions import FUNCTIONS

PLOT_DIR = os.path.join(ROOT_DIR, "function_plots")

SINGLE_PLOT = False
DIMENSIONS = 2
RESOLUTION = 5000
NORMALIZED = True

DOMAIN = tf.convert_to_tensor([[-1, -1], [1, 1]], dtype=tf.float32)
X = tf.range(DOMAIN[0][0], DOMAIN[1][0], (DOMAIN[1][0] - DOMAIN[0][0]) / RESOLUTION)
Y = tf.range(DOMAIN[0][1], DOMAIN[1][1], (DOMAIN[1][1] - DOMAIN[0][1]) / RESOLUTION)
MESHGRID = tf.convert_to_tensor(tf.meshgrid(X, Y))


def main():
    function_idx = 0 if NORMALIZED else 2
    fig_name = "norm_" if NORMALIZED else ""

    if SINGLE_PLOT:
        for function_name in FUNCTIONS.keys():
            image = FUNCTIONS[function_name][function_idx](MESHGRID)
            print("plotting {}, max={}, min={}, area={}".format(function_name,
                                                                tf.reduce_max(image),
                                                                tf.reduce_min(image),
                                                                tf.reduce_mean(image)))
            fig, ax = plt.subplots(1, 1)
            p = plot(image, ax)
            plt.colorbar(p)
            plt.savefig(os.path.join(PLOT_DIR, "{}{}_{}D".format(fig_name, function_name, DIMENSIONS)),
                        dpi=400, transparent=True)
            plt.show()
    else:
        images = []
        for function_name in FUNCTIONS.keys():
            images.append(FUNCTIONS[function_name][function_idx](MESHGRID))
        sub_plots(images)
        plt.savefig(os.path.join(PLOT_DIR, "{}All_{}D".format(fig_name, DIMENSIONS)),
                    dpi=400, transparent=True)
        plt.show()


def plot(image, ax):
    if DIMENSIONS == 3:
        ax.set_yticklabels([])
        ax.set_xticklabels([])
        ax.set_zticklabels([])
        return ax.plot_surface(MESHGRID[0], MESHGRID[1], image, cmap="viridis")
    elif DIMENSIONS == 2:
        extent = (DOMAIN[0, 0], DOMAIN[1, 0], DOMAIN[0, 1], DOMAIN[1, 1])
        levels = [-1, -0.75, -0.5, -0.25, 0.25, 0.5, 0.75, 0.85, 0.95, 0.999]

        ax.contour(MESHGRID[0], MESHGRID[1], image, levels, colors='k', alpha=1.0, extent=extent, linewidths=0.5)
        ax.contour(MESHGRID[0], MESHGRID[1], image, [0, ], colors='k', extent=extent, linewidths=1.00,
                   linestyles='solid')
        return ax.imshow(np.flipud(image), cmap="viridis", extent=[-1.0, 1.0, -1.0, 1.0])


def sub_plots(images):
    fig = plt.figure()
    fig.set_size_inches(7, 10)
    plt.subplots_adjust(left=0.05, bottom=0.05, right=0.95, top=0.95, wspace=0.05, hspace=0.2)
    finished = []
    for i, (label, image) in enumerate(zip(FUNCTIONS.keys(), images)):
        timestamp = time.strftime('%H:%M:%S', time.gmtime((time.time() - start_time)))
        print("{} finished, {}".format(",".join(finished), timestamp), end="")
        if DIMENSIONS == 3:
            ax = fig.add_subplot(2, 4, i + 1, projection='3d')
        elif DIMENSIONS == 2:
            ax = fig.add_subplot(4, 2, i + 1)
        else:
            raise NotImplementedError
        if i < 6:
            ax.set_xticklabels([])
        else:
            ax.set_xticklabels([-1, None, 0, None, 1])
        if i % 2 == 1:
            ax.set_yticklabels([])
        else:
            ax.set_yticklabels([-1, None, 0, None, 1])

        label = label.capitalize()
        ax.set_title(label, fontsize=10)
        label = label.lower()
        p = plot(image, ax)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='5%', pad=0.1)
        fig.colorbar(p, cax=cax, orientation='vertical')
        finished.append(label)
        print(end="\r")


if __name__ == "__main__":
    start_time = time.time()
    if not os.path.isdir(PLOT_DIR):
        os.makedirs(PLOT_DIR)
    main()
