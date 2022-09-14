import math
import os

from matplotlib import pyplot as plt

from objective_functions.tf_objective_functions import FUNCTIONS
from common.utils import create_rotation_matrix
from plot_function import PLOT_DIR
import tensorflow as tf

domain = tf.convert_to_tensor([[-1, -1], [1, 1]], dtype=tf.float32)

n = 2500
X = tf.range(domain[0][0], domain[1][0], (domain[1][0] - domain[0][0]) / n)
Y = tf.range(domain[0][1], domain[1][1], (domain[1][1] - domain[0][1]) / n)
meshgrid = tf.convert_to_tensor(tf.meshgrid(X, Y))

extent = (domain[0, 0], domain[1, 0], domain[0, 1], domain[1, 1])
levels = [-1, -0.75, -0.5, -0.25, 0.25, 0.5, 0.75, 0.85, 0.95, 0.999]

function_name = "ackley"

def plot_category(axes, _Z, contour):
    plt.xticks([])
    plt.yticks([])
    axes.imshow(_Z, cmap="viridis", extent=[-1.0, 1.0, 1.0, -1.0])
    if contour:
        axes.contour(meshgrid[0], meshgrid[1], _Z, levels, colors='k', alpha=1.0, extent=extent, linewidths=0.5)
        axes.contour(meshgrid[0], meshgrid[1], _Z, [0, ], colors='k', extent=extent, linewidths=0.75, linestyles='solid')
    else:
        axes.contour(meshgrid[0], meshgrid[1], _Z, levels, colors='k', alpha=0.1, extent=extent, linewidths=0.1)
        axes.contour(meshgrid[0], meshgrid[1], _Z, [0, ], colors='k', extent=extent, alpha=0.4, linewidths=0.25, linestyles='solid')


translated_mesh = meshgrid - 0.7

Z = FUNCTIONS[function_name][0](translated_mesh)
fig = plt.figure()
plt.subplots_adjust(left=0.05, bottom=0.05, right=0.95, top=0.95, wspace=0.05, hspace=0.05)
fig.set_size_inches(3, 10)
ax = fig.add_subplot(4, 1, 1)
plot_category(ax, Z, True)

rotation = create_rotation_matrix(math.pi / 4)
rotated_mesh = [
    meshgrid[0] * rotation[0][0] +
    meshgrid[1] * rotation[1][0],
    meshgrid[0] * rotation[0][1] +
    meshgrid[1] * rotation[1][1]
]
Z = FUNCTIONS[function_name][0](rotated_mesh)
ax = fig.add_subplot(4, 1, 2)
plot_category(ax, Z, True)

noisy_meshgrid = tf.add(meshgrid, tf.random.uniform(meshgrid.shape, minval=-0.05, maxval=0.05, dtype=tf.float32))
Z = FUNCTIONS[function_name][0](noisy_meshgrid)
ax = fig.add_subplot(4, 1, 3)
plot_category(ax, Z, False)



Z = FUNCTIONS[function_name][0](meshgrid)
Z += tf.random.uniform(Z.shape, minval=-0.05, maxval=0.05, dtype=tf.float32)
ax = fig.add_subplot(4, 1, 4)

plot_category(ax, Z, False)
plt.savefig(os.path.join(PLOT_DIR, "operations_{}.png".format(function_name)), transparent=True)
plt.show()
