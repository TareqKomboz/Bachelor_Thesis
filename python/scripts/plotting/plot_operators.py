import math
import os

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

from common.utils import create_rotation_matrix
from definitons import ROOT_DIR
from objective_functions.tf_objective_functions import FUNCTIONS

translation = (-0.8, 0.8)
rotation = math.pi / 4
in_noise = 0.015
out_noise = 0.015
function = FUNCTIONS["ackley"][0]
labels = ("Translation", "Rotation", "Input noise", "Output noise")

RESOLUTION = 640
DOMAIN = tf.convert_to_tensor([[-1, -1], [1, 1]], dtype=tf.float32)
X = tf.range(DOMAIN[0][0], DOMAIN[1][0], (DOMAIN[1][0] - DOMAIN[0][0]) / RESOLUTION)
Y = tf.range(DOMAIN[0][1], DOMAIN[1][1], (DOMAIN[1][1] - DOMAIN[0][1]) / RESOLUTION)
MESHGRID = tf.convert_to_tensor(tf.meshgrid(X, Y))

def main():
    images = draw_images()
    fig, axs = plt.subplots(1, 4)
    fig.set_size_inches(10, 3)
    plt.subplots_adjust(wspace=0.3, hspace=0.3)
    plt.gcf().subplots_adjust(right=0.85)
    for i, (ax, image) in enumerate(zip(axs, images)):
        p = plot(ax, image)
        ax.title.set_text(labels[i])
        if i > 0:
            ax.set_yticklabels([])
        else:
            ax.set_yticklabels([-1, None, 0, None, 1])
        ax.set_xticklabels([-1, 0, 1])
    # divider = make_axes_locatable(ax)
    cax = plt.gcf().add_axes([0.87, 0.25, 0.015, 0.48])
    plt.colorbar(p, cax=cax, orientation='vertical')
    plt.savefig(os.path.join(ROOT_DIR, 'function_plots', 'operator'), dpi=400, transparent=True)
    plt.show()

def draw_images():
    translated_mesh = tf.convert_to_tensor((
        tf.subtract(MESHGRID[0], translation[0]),
        tf.subtract(MESHGRID[1], translation[1])
    ))
    images = [function(translated_mesh)]
    print("translation", end="")
    R = create_rotation_matrix(rotation)
    rotated_mesh = tf.convert_to_tensor((
        MESHGRID[0] * R[0][0] +
        MESHGRID[1] * R[0][1],
        MESHGRID[0] * R[1][0] +
        MESHGRID[1] * R[1][1]
    ))
    images.append(function(rotated_mesh))
    print(", rotation", end="")
    in_noisy_mesh = MESHGRID + tf.random.normal(MESHGRID.shape, mean=0.0, stddev=in_noise)
    images.append(function(in_noisy_mesh))
    print(", input_noise", end="")
    image = function(MESHGRID)
    image += tf.random.normal(image.shape, mean=0.0, stddev=out_noise)
    images.append(image)
    print(", output_noise drawn")
    return images


def plot(ax, image):
    extent = [-1.0, 1.0, -1.0, 1.0]
    levels = [-1, -0.75, -0.5, -0.25, 0.25, 0.5, 0.75, 0.85, 0.95, 0.999]

    ax.contour(MESHGRID[0], MESHGRID[1], image, levels, colors='k', alpha=1.0, extent=extent, linewidths=0.5)
    ax.contour(MESHGRID[0], MESHGRID[1], image, [0, ], colors='k', extent=extent, linewidths=0.75,
               linestyles='solid')
    return ax.imshow(np.flipud(image), cmap="viridis", extent=extent)

if __name__ == "__main__":
    main()
