import math
import tensorflow as tf

# All functions must take inputs of range (-1, 1) of dimension (2, N),
# the normalized version output is scaled so the surface integral equals 0 and the global minimum is 1
# global maximum clipped to -1 if necessary

# norm = (max - f) / (max - min)

dtype = tf.float32


@tf.function
def ackley(x):
    x = tf.multiply(x, 32.0)
    a = tf.constant(20.0, dtype=dtype)  # 10
    b = tf.constant(0.2, dtype=dtype)
    c = tf.constant(2.0 * math.pi, dtype=dtype)  # 0.25
    d = x.shape[0]

    sum1 = 0.0
    for i in range(d):
        sum1 += tf.pow(x[i], 2)

    sum2 = 0.0
    for j in range(d):
        sum2 += tf.cos(c * x[j])

    f = (-a * tf.exp(-b * tf.sqrt((1 / d) * sum1))) - tf.exp((1 / d) * sum2) + tf.exp(1.0) + a
    return f


@tf.function
def norm_ackley(x):
    my_max = ackley(x=tf.ones_like(input=x))
    value = my_max - ackley(x)
    value /= my_max
    value = tf.clip_by_value(value, clip_value_min=-1.0, clip_value_max=1.0)
    return value


def ackley_gradient(x):
    a = tf.constant(20.0, dtype=dtype)  # 10
    b = tf.constant(0.2, dtype=dtype)
    c = tf.constant(2.0 * math.pi, dtype=dtype)  # 0.25
    d = x.shape[0]
    gradient = []

    sum1 = 0.0
    for i in range(d):
        sum1 += tf.pow(x[i], 2)

    sum2 = 0.0
    for j in range(d):
        sum2 += tf.cos(c * x[j])

    for k in range(d):
        gradient.append((a * b * x[k] * tf.exp(-b * tf.sqrt((1 / d) * sum1)) / tf.sqrt((d * sum1))) + (tf.exp((1 / d) * sum2) * (c / d) * tf.sin(c * x[k])))

    return gradient


# plot for x between 5, 10, 50 and 1000
# f(x)=0 for 000
# @tf.function
def griewank(x):
    x = tf.multiply(x, 600.0)
    d = x.shape[0]

    my_sum = 0.0
    for i in range(d):
        my_sum += tf.math.divide(tf.pow(x[i], 2.0), 4000.0)

    prod = 1.0
    for j in range(d):
        prod = tf.multiply(prod, tf.cos(tf.math.divide(x[j], tf.math.sqrt(tf.cast(j + 1, dtype=tf.float32)))))

    f = tf.math.add(tf.math.subtract(my_sum, prod), 1.0)

    return f


# @tf.function
def norm_griewank(x):
    my_max = griewank(x=tf.ones_like(input=x))
    value = my_max - griewank(x)
    value /= my_max
    value = tf.clip_by_value(value, clip_value_min=-1.0, clip_value_max=1.0)
    return value


def griewank_gradient(x):
    d = x.shape[0]
    gradient = []
    for i in range(d):
        casted_i = tf.cast(i + 1, dtype=tf.float32)
        gradient.append((x[i] / 2000.0) + ((1 / tf.sqrt(casted_i)) * tf.sin(tf.math.divide(x[i], tf.math.sqrt(casted_i)))))

    return gradient


# @tf.function
# def langermann(x):
#     x += 1
#     x *= 5.0
#
#     m = 5
#     c = tf.constant([1, 2, 5, 2, 3], dtype=dtype)
#     a = tf.constant([[3, 5, 2, 1, 7], [5, 2, 1, 4, 9]], dtype=dtype)
#     d = x.shape[0]
#     batch_size = x.shape[1]
#
#     f = tf.zeros(shape=(batch_size,))
#     for i in tf.range(m):
#         my_sum = tf.zeros(shape=(batch_size,))
#         for j in tf.range(d):
#             my_sum += tf.pow(tf.subtract(x[j], a[j][i]), 2)
#
#         f += c[i] * tf.multiply(tf.exp(-(1.0 / math.pi) * my_sum), tf.cos(math.pi * my_sum))
#
#     return f


# @tf.function
# def norm_langermann(x):
#     value = 5.0 - langermann(x)
#     value /= 10.0
#     value = tf.clip_by_value(value, clip_value_min=-1.0, clip_value_max=1.0)
#     return value


# f(x)=0 for 111
@tf.function
def levy(x):
    x = tf.multiply(x, 10.0)
    d = x.shape[0]

    term1 = (tf.sin(math.pi * w(x[0]))) ** 2
    term3 = ((w(x[d-1]) - 1) ** 2) * (1 + (tf.sin(2 * math.pi * w(x[d-1]))) ** 2)
    my_sum = 0
    for i in range(d - 1):
        my_sum += ((w(x[i]) - 1) ** 2) * (1 + (10 * (tf.sin((math.pi * w(x[i])) + 1) ** 2)))

    f = term1 + my_sum + term3

    return f


@tf.function
def norm_levy(x):
    my_max = levy(x=-tf.ones_like(input=x))
    value = my_max - levy(x)
    value /= my_max
    value = tf.clip_by_value(value, clip_value_min=-1.0, clip_value_max=1.0)
    return value


def levy_gradient(x):
    d = x.shape[0]
    gradient = []
    for i in range(d):
        if i == 0:
            term1 = 0.5 * tf.sin(math.pi * w(x[0])) * tf.cos(math.pi * w(x[0])) * math.pi
            term2 = 0.5 * (w(x[i]) - 1) * (1 + 10 * (tf.sin(math.pi * w(x[i]) + 1) ** 2)) + \
                    5 * math.pi * ((w(x[i]) - 1) ** 2) * tf.sin((math.pi * w(x[i])) + 1) * tf.cos((math.pi * w(x[i])) + 1)
            gradient.append(term1 + term2)
        elif i == d - 1:
            term3 = 0.5 * (w(x[i]) - 1) * (1 + (tf.sin(2 * math.pi * w(x[i])) ** 2)) + \
                    math.pi * ((w(x[i]) - 1) ** 2) * tf.sin(2 * math.pi * w(x[i])) * tf.cos(2 * math.pi * w(x[i]))
            gradient.append(term3)
        else:
            term2 = 0.5 * (w(x[i]) - 1) * (1 + 10 * (tf.sin(math.pi * w(x[i]) + 1) ** 2)) + \
                    5 * math.pi * ((w(x[i]) - 1) ** 2) * tf.sin((math.pi * w(x[i])) + 1) * tf.cos(
                (math.pi * w(x[i])) + 1)
            gradient.append(term2)

    return gradient


def w(x_i):
    return 1 + ((x_i - 1) / 4)


# @tf.function
# def michalewicz(x):
#     x *= 0.5 * math.pi
#     x += 0.5 * math.pi
#
#     m = 10
#     d = x.shape[0]
#     f = 0.0
#     for i in range(d):
#         f += tf.sin(x[i]) * (tf.sin((((i + 1) * (x[i] ** 2)) / math.pi)) ** (2 * m))
#     return -f
#
#
# @tf.function
# def norm_michalewicz(x):
#     value = - 0.20895 - michalewicz(x)
#     value /= 1.592327
#     value = tf.clip_by_value(value, clip_value_min=-1.0, clip_value_max=1.0)
#     return value


@tf.function
def rastrigin(x):
    x = tf.multiply(x, 5.12)
    pi2 = tf.constant(2 * math.pi, dtype=tf.float32)
    a = tf.constant(10.0, dtype=tf.float32)

    d = x.shape[0]
    my_sum = 0.0
    for i in range(d):
        my_sum += (tf.pow(x[i], 2) - (a * tf.cos(pi2 * x[i])))

    f = a * d + my_sum

    return f


@tf.function
def norm_rastrigin(x):
    my_max = rastrigin(x=tf.ones_like(input=x))  # Todo: max = 37.0511589 or 80
    value = my_max - rastrigin(x)
    value /= my_max
    value = tf.clip_by_value(value, clip_value_min=-1.0, clip_value_max=1.0)
    return value


def rastrigin_gradient(x):
    d = x.shape[0]
    gradient = []
    for i in range(d):
        gradient.append((2 * x[i]) + (10 * 2 * math.pi * tf.sin(2 * math.pi * x[i])))

    return gradient


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
    my_max = tf.math.log(rosenbrock(x=-tf.ones_like(input=x)))
    value = my_max - tf.math.log(rosenbrock(x))  # 5 - ...
    value /= my_max  # 20
    value = tf.clip_by_value(value, clip_value_min=-1.0, clip_value_max=1.0)
    return value


@tf.function
def norm_rosenbrock(x):
    my_max = rosenbrock(x=-tf.ones_like(input=x))  # my_max = 3609.0
    value = my_max - rosenbrock(x)
    value /= my_max
    value = tf.clip_by_value(value, clip_value_min=-1.0, clip_value_max=1.0)
    return value


def rosenbrock_gradient(x):
    d = x.shape[0]
    gradient = []
    for i in range(d):
        if i == 0:
            gradient.append((-400 * x[i] * (x[i+1] - (x[i] ** 2))) + 2 * (x[i] - 1))
        elif i == d - 1:
            gradient.append((-400 * x[i] * (x[i+1] - (x[i] ** 2))) + (2 * (x[i] - 1)) + (200 * (x[i] - (x[i-1] ** 2))))
        else:
            gradient.append(200 * (x[i] - (x[i-1] ** 2)))

    return gradient


@tf.function
def sphere(x):
    x *= 10.0
    x = tf.pow(x, 2)
    return tf.math.reduce_sum(x, axis=0)


@tf.function
def norm_sphere(x):
    my_max = sphere(x=tf.ones_like(input=x))
    value = my_max - sphere(x)
    value /= my_max
    value = tf.clip_by_value(value, clip_value_min=-1.0, clip_value_max=1.0)
    return value


def sphere_gradient(x):
    d = x.shape[0]
    gradient = []
    for i in range(d):
        gradient.append(2*x[i])

    return gradient


@tf.function
def styblinski_tang(x):
    x *= 5.0
    d = x.shape[0]

    my_sum = 0.0
    for i in range(d):
        my_sum += (x[i] ** 4) - (16 * (x[i] ** 2)) + (5.0 * x[i])

    f = 0.5 * my_sum

    return f


@tf.function
def norm_styblinski_tang(x):
    d = x.shape[0]
    my_max = styblinski_tang(x=tf.ones_like(input=x))
    value = my_max - styblinski_tang(x)
    value /= (my_max + (39.16599 * d))
    value = tf.clip_by_value(value, clip_value_min=-1.0, clip_value_max=1.0)
    return value


def styblinski_tang_gradient(x):
    d = x.shape[0]
    gradient = []
    for i in range(d):
        gradient.append(0.5 * ((4 * (x[i] ** 3)) - (32 * x[i]) + 5))

    return gradient
    
    
# f(x)=0 bei 000
@tf.function
def zakharov(x):
    x = (((x + 1.0) / 2.0) * 15.0) - 5.0
    d = x.shape[0]

    sum1 = 0.0
    for i in range(d):
        sum1 += x[i] ** 2

    sum2 = 0.0
    for j in range(d):
        sum2 += 0.5 * (j + 1) * x[j]

    sum3 = 0.0
    for k in range(d):
        sum3 += 0.5 * (k + 1) * x[k]

    f = sum1 + (sum2 ** 2) + (sum3 ** 4)

    return f


@tf.function
def norm_zakharov(x):
    my_max = zakharov(x=tf.ones_like(input=x))
    value = my_max - zakharov(x)
    value /= my_max
    value = tf.clip_by_value(value, clip_value_min=-1.0, clip_value_max=1.0)
    return value


def zakharov_gradient(x):
    d = x.shape[0]
    gradient = []

    sum1 = 0.0
    for j in range(d):
        sum1 += (j + 1) * x[j]

    sum2 = 0.0
    for k in range(d):
        sum2 += 0.5 * (k + 1) * x[k]

    for i in range(d):
        gradient.append((2 * x[i]) + (0.5 * i * sum1) + (2 * i * (sum2 ** 3)))

    return gradient




FUNCTIONS = {
    "ackley": (norm_ackley, norm_ackley, ackley),
    "griewank": (norm_griewank, norm_griewank, griewank),
    # "langermann": (norm_langermann, norm_langermann, langermann),
    "levy": (norm_levy, norm_levy, levy),
    # "michalewicz": (norm_michalewicz, norm_michalewicz, michalewicz),
    "rastrigin": (norm_rastrigin, norm_rastrigin, rastrigin),
    "rosenbrock": (lognorm_rosenbrock, lognorm_rosenbrock, rosenbrock),
    "sphere": (norm_sphere, norm_sphere, sphere),
    "styblinski_tang": (norm_styblinski_tang, norm_styblinski_tang, styblinski_tang),
    "zakharov": (norm_zakharov, norm_zakharov, zakharov)
}

"""
FUNCTIONS = {
    "rosenbrock": (lognorm_rosenbrock, lognorm_rosenbrock, rosenbrock)
}
"""
