import math

import numpy as np
import tensorflow as tf

# All functions must take inputs of range (-1, 1) of dimension (2, N),
# the normalized version output is scaled so the surface integral equals 0 and the global minimum is 1
# global maximum clipped to -1 if necessary
from scipy.optimize import optimize


@tf.function
def ackley2d(x):
    dtype = tf.float32
    x = tf.multiply(x, 32.0)
    a = tf.constant(10, dtype=dtype)
    b = tf.constant(0.2, dtype=dtype)
    c = tf.constant(0.25 * math.pi, dtype=dtype)
    value = (-a * tf.exp(-b * tf.sqrt(0.5 * (tf.pow(x[0], 2) + tf.pow(x[1], 2)))) - tf.exp(
        0.5 * (tf.cos(c * x[0]) + tf.cos(c * x[1]))) + tf.exp(1.0) + a)
    return value


# local minimum (1, 1)
@tf.function
def norm_ackley2d(x):
    value = 10.852 - ackley2d(x)
    return tf.divide(value, 10.85)


@tf.function
def cross_in_tray(x):
    x = tf.multiply(x, 15.0)
    x = tf.cast(x, tf.float64)
    a = tf.abs(100 - tf.sqrt(x[0] * x[0] + x[1] * x[1]) / math.pi)
    b = tf.abs(tf.sin(x[0]) * tf.sin(x[1]) * tf.exp(a)) + 1
    c = -0.0001 * b ** 0.1
    return tf.cast(c, tf.float32)


# local minimum
@tf.function
def norm_cross_in_tray(x):
    value = - 1.3604403734207153 - cross_in_tray(x)
    value = tf.multiply(value, 1.43)
    value = tf.clip_by_value(value, clip_value_min=-1.0, clip_value_max=1.0)
    return value


@tf.function
def quadratic(x):
    x = tf.pow(x, 2)
    return -(x[0] + x[1])


@tf.function
def norm_quadratic(x):
    value = 0.66666 - quadratic(x)
    value = tf.multiply(value, 1.500150015)
    value = tf.clip_by_value(value, clip_value_min=-1.0, clip_value_max=1.0)
    return value / 1.5


@tf.function
def rastrigin2D(x):
    x = tf.convert_to_tensor(x, dtype=tf.float32)
    a = tf.constant(10.0, dtype=tf.float32)
    x *= 5.12
    pi2 = 2 * math.pi
    return 2 * a + x[0] ** 2 - a * tf.cos(pi2 * x[0]) + x[1] ** 2 - a * tf.cos(pi2 * x[1])


@tf.function
def norm_rastrigin2D(x):
    value = 37.0511589 - rastrigin2D(x)
    value /= 37.0511589
    value = tf.clip_by_value(value, clip_value_min=-1.0, clip_value_max=1.0)
    return value


@tf.function
def rosenbrock(x):
    a = 100.0
    x *= 2.0
    return a * (x[1] - x[0] ** 2) ** 2 + (1 - x[0]) ** 2


@tf.function
def plot_rosenbrock(x):
    a = rosenbrock(x) + math.e
    return tf.math.log(a) - 1


# local minimum (0,0) = 0,
@tf.function
def lognorm_rosenbrock(x):
    value = 5.0 - tf.math.log(rosenbrock(x))
    value /= 20
    value = tf.clip_by_value(value, clip_value_min=-1.0, clip_value_max=1.0)
    return value


@tf.function
def norm_rosenbrock(x):
    a = 3609.0 - rosenbrock(x)
    a /= 3609.0
    return a


@tf.function
def himmelblau(x):
    x *= 6
    a = (x[0] ** 2 + x[1] - 11) ** 2
    b = (x[0] + x[1] ** 2 - 7) ** 2
    return a + b


# local minimum (0,0)
@tf.function
def norm_himmelblau(x):
    value = 5 - tf.math.log(himmelblau(x))
    value /= 7.3
    value = tf.clip_by_value(value, clip_value_min=-1.0, clip_value_max=1.0)
    return value


@tf.function
def plot_himmelblau(x):
    return tf.math.log(himmelblau(x))


@tf.function
def zakharov(x):
    x *= 10
    return quadratic(x) + (0.5 * x[0] + x[1]) ** 2 + (0.5 * x[0] + x[1]) ** 4


@tf.function
def norm_zakharov(x):
    value = 3900 - zakharov(x)
    value /= 3900
    value = tf.clip_by_value(value, clip_value_min=-1.0, clip_value_max=1.0)
    return value


@tf.function
def booth(x):
    x *= 10
    return (x[0] + 2 * x[1] - 7) ** 2 + (2 * x[0] + x[1] - 5) ** 2


@tf.function
def norm_booth(x):
    value = 408.4613952636719 - booth(x)
    value /= 408.4613952636719
    value = tf.clip_by_value(value, clip_value_min=-1.0, clip_value_max=1.0)
    return value


@tf.function
def mccormick(x):
    xx = x[0] * 2.75
    xx -= 2.75
    yy = x[1] * 3.5
    yy -= 2.5
    return tf.sin(xx + yy) + (xx - yy) ** 2 - 1.5 * xx + 2.5 * yy + 1


@tf.function
def norm_mccormick(x):
    value = 4.522 - mccormick(x)
    value /= 9.5767
    value = tf.clip_by_value(value, clip_value_min=-1.0, clip_value_max=1.0)
    return value


@tf.function
def camel(x):
    xx = x[0] * 2
    yy = x[1] * 1
    a = (4 - 2.1 * xx ** 2 + (xx ** 4 / 3)) * xx ** 2
    b = (-4 + 4 * yy ** 2) * yy ** 2
    return a + xx * yy + b


@tf.function
def norm_camel(x):
    value = 1.12765 - camel(x)
    value /= 2.159264
    value = tf.clip_by_value(value, clip_value_min=-1.0, clip_value_max=1.0)
    return value


@tf.function
def michalewicz(x):
    x *= 0.5 * math.pi
    x += 0.5 * math.pi
    a = tf.sin(x[0]) * tf.sin(x[0] ** 2 / math.pi) ** 20
    b = tf.sin(x[1]) * tf.sin(2 * x[1] ** 2 / math.pi) ** 20
    return - (a + b)


@tf.function
def norm_michalewicz(x):
    value = - 0.20895 - michalewicz(x)
    value /= 1.592327
    value = tf.clip_by_value(value, clip_value_min=-1.0, clip_value_max=1.0)
    return value

def plane(x):
    return (x[1] / 4.95) + 0.2


def zero(x):
    return tf.zeros_like(x)[0]


FUNCTIONS = {
    "ackley": (norm_ackley2d, norm_ackley2d, ackley2d),
    "himmelblau": (norm_himmelblau, norm_himmelblau, himmelblau),
    "cross-in-tray": (norm_cross_in_tray, norm_cross_in_tray, cross_in_tray),
    "rastrigin": (norm_rastrigin2D, norm_rastrigin2D, rastrigin2D),
    "sphere": (norm_quadratic, norm_quadratic, quadratic),
    "camel": (norm_camel, norm_camel, camel),
    "rosenbrock": (lognorm_rosenbrock, lognorm_rosenbrock, rosenbrock),
    "michalewicz": (norm_michalewicz, norm_michalewicz, michalewicz)
}
"""
FUNCTIONS = {

    "rosenbrock": (lognorm_rosenbrock, lognorm_rosenbrock, rosenbrock)

}
"""
