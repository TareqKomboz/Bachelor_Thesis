import math
import tensorflow as tf

# All functions must take inputs of range (-1, 1) of dimension (2, N),
# the normalized version output is scaled so the surface integral equals 0 and the global minimum is 1
# global maximum clipped to -1 if necessary


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


# @tf.function
def langermann(x):
    dtype = tf.float32
    m = 5
    c = tf.constant([1, 2, 5, 2, 3], dtype=dtype)
    a = tf.constant([[3, 5, 2, 1, 7], [5, 2, 1, 4, 9]], dtype=dtype)
    d = x.shape[0]

    f = 0.0
    for i in tf.range(m):
        summe = 0.0
        for j in tf.range(d):
            summe += tf.pow(tf.subtract(x[j], a[j][i]), 2)

        f += c[i] * tf.multiply(tf.exp(-(1.0 / math.pi) * summe), tf.cos(math.pi * summe))

    return f


# local minimum (1, 1)
# @tf.function
def norm_langermann(x):
    return langermann(x) / 5.0


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


@tf.function
def rastrigin2d(x):
    x = tf.convert_to_tensor(x, dtype=tf.float32)
    a = tf.constant(10.0, dtype=tf.float32)
    x *= 5.12
    pi2 = 2 * math.pi
    return 2 * a + x[0] ** 2 - a * tf.cos(pi2 * x[0]) + x[1] ** 2 - a * tf.cos(pi2 * x[1])


@tf.function
def norm_rastrigin2d(x):
    value = 37.0511589 - rastrigin2d(x)
    value /= 37.0511589
    value = tf.clip_by_value(value, clip_value_min=-1.0, clip_value_max=1.0)
    return value


@tf.function
def rosenbrock(x):
    x *= 2.0
    return 100.0 * ((x[1] - (x[0] ** 2)) ** 2) + ((x[0] - 1) ** 2)


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
def sumsquares(x):
    x = tf.pow(x, 2)
    return tf.math.reduce_sum(x, axis=0)


@tf.function
def norm_sumsquares(x):
    return -sumsquares(x)


FUNCTIONS = {
    "ackley": (norm_ackley2d, norm_ackley2d, ackley2d),
    "langermann": (norm_langermann, norm_langermann, langermann),
    "michalewicz": (norm_michalewicz, norm_michalewicz, michalewicz),
    "rastrigin": (norm_rastrigin2d, norm_rastrigin2d, rastrigin2d),
    "rosenbrock": (lognorm_rosenbrock, lognorm_rosenbrock, rosenbrock),
    "sumsquares": (norm_sumsquares, norm_sumsquares, sumsquares)
}
"""
FUNCTIONS = {

    "rosenbrock": (lognorm_rosenbrock, lognorm_rosenbrock, rosenbrock)

}
"""
