import math
import tensorflow as tf

# All functions must take inputs of range (-1, 1) of dimension (2, N),
# the normalized version output is scaled so the surface integral equals 0 and the global minimum is 1
# global maximum clipped to -1 if necessary

dtype = tf.float32


@tf.function
def ackley(x):
    x = tf.multiply(x, 32.0)
    a = tf.constant(10.0, dtype=dtype)
    b = tf.constant(0.2, dtype=dtype)
    c = tf.constant(0.25 * math.pi, dtype=dtype)
    d = x.shape[0]

    sum1 = 0
    for i in range(d):
        sum1 += tf.pow(x[i], 2)

    sum2 = 0
    for j in range(d):
        sum2 += tf.cos(c * x[j])

    f = (-a * tf.exp(-b * tf.sqrt((1 / d) * sum1))) - tf.exp((1 / d) * sum2) + tf.exp(1.0) + a
    return f


# local minimum (1, 1)
@tf.function
def norm_ackley(x):
    value = 10.852 - ackley(x)
    value /= 10.852
    value = tf.clip_by_value(value, clip_value_min=-1.0, clip_value_max=1.0)
    return value


@tf.function
def langermann(x):
    x += 1
    x *= 5.0

    m = 5
    c = tf.constant([1, 2, 5, 2, 3], dtype=dtype)
    a = tf.constant([[3, 5, 2, 1, 7], [5, 2, 1, 4, 9]], dtype=dtype)
    d = x.shape[0]
    batch_size = x.shape[1]

    f = tf.zeros(shape=(batch_size,))
    for i in tf.range(m):
        summe = tf.zeros(shape=(batch_size,))
        for j in tf.range(d):
            summe += tf.pow(tf.subtract(x[j], a[j][i]), 2)

        f += c[i] * tf.multiply(tf.exp(-(1.0 / math.pi) * summe), tf.cos(math.pi * summe))

    return f


# local minimum (1, 1)
@tf.function
def norm_langermann(x):
    value = 5.0 - langermann(x)
    value /= 10.0
    value = tf.clip_by_value(value, clip_value_min=-1.0, clip_value_max=1.0)
    return value


@tf.function
def michalewicz(x):
    x *= 0.5 * math.pi
    x += 0.5 * math.pi

    m = 10
    d = x.shape[0]
    f = 0.0
    for i in range(d):
        f += tf.sin(x[i]) * (tf.sin((((i + 1) * (x[i] ** 2)) / math.pi)) ** (2 * m))
    return -f


@tf.function
def norm_michalewicz(x):
    value = - 0.20895 - michalewicz(x)
    value /= 1.592327
    value = tf.clip_by_value(value, clip_value_min=-1.0, clip_value_max=1.0)
    return value


@tf.function
def rastrigin(x):
    x = tf.convert_to_tensor(x, dtype=tf.float32)
    x *= 5.12
    pi2 = 2 * math.pi
    a = tf.constant(10.0, dtype=tf.float32)

    d = x.shape[0]
    sum = 0
    for i in range(d):
        sum += ((x[i] ** 2) - (a * tf.cos(pi2 * x[i])))

    f = a * d + sum

    return f


@tf.function
def norm_rastrigin(x):
    value = 37.0511589 - rastrigin(x)  # Todo: 80
    value /= 37.0511589
    value = tf.clip_by_value(value, clip_value_min=-1.0, clip_value_max=1.0)
    return value


@tf.function
def rosenbrock(x):
    x *= 2.0  # Todo x in (-5.0, 10.0) or (-2.048, 2.048)
    d = x.shape[0]

    f = 0
    for i in range(d - 1):
        f += (100.0 * ((x[i + 1] - (x[i] ** 2)) ** 2)) + ((x[i] - 1) ** 2)

    return f


# local minimum (0,0) = 0,
@tf.function
def lognorm_rosenbrock(x):
    value = 5.0 - tf.math.log(rosenbrock(x))
    value /= 20
    value = tf.clip_by_value(value, clip_value_min=-1.0, clip_value_max=1.0)
    return value


@tf.function
def norm_rosenbrock(x):
    value = 3609.0 - rosenbrock(x)
    value /= 3609.0
    value = tf.clip_by_value(value, clip_value_min=-1.0, clip_value_max=1.0)
    return value


@tf.function
def sumsquares(x):
    x *= 10.0
    x = tf.pow(x, 2)
    return tf.math.reduce_sum(x, axis=0)


@tf.function
def norm_sumsquares(x):
    value = 200.0 - sumsquares(x)
    value /= 200.0
    value = tf.clip_by_value(value, clip_value_min=-1.0, clip_value_max=1.0)
    return value


FUNCTIONS = {
    "ackley": (norm_ackley, norm_ackley, ackley),
    # "langermann": (norm_langermann, norm_langermann, langermann),
    "michalewicz": (norm_michalewicz, norm_michalewicz, michalewicz),
    "rastrigin": (norm_rastrigin, norm_rastrigin, rastrigin),
    "rosenbrock": (lognorm_rosenbrock, lognorm_rosenbrock, rosenbrock),
    "sumsquares": (norm_sumsquares, norm_sumsquares, sumsquares)
}
"""
FUNCTIONS = {
    "rosenbrock": (lognorm_rosenbrock, lognorm_rosenbrock, rosenbrock)
}
"""
