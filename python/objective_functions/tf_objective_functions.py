import math
import tensorflow as tf

# All functions must take inputs of range (-1, 1) of dimension (2, N),
# the normalized version output is scaled so the surface integral equals 0 and the global minimum is 1
# global maximum clipped to -1 if necessary

# norm = (max - f) / (max - min)

dtype = tf.float32


# f(x)=0 at 000
@tf.function
def ackley(x):
    x = tf.multiply(x, 20.0)  # 32
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


# plot for x between 5, 10, 50 and 1000
# f(x)=0 for 000
@tf.function
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


# f(x)=0 at 000
@tf.function
def rastrigin(x):
    x = tf.multiply(x, 5.0)
    pi2 = tf.constant(2 * math.pi, dtype=tf.float32)
    a = tf.constant(10.0, dtype=tf.float32)

    d = x.shape[0]
    my_sum = 0.0
    for i in range(d):
        my_sum += (tf.pow(x[i], 2) - (a * tf.cos(pi2 * x[i])))

    f = a * d + my_sum

    return f


@tf.function
def rosenbrock(x):
    x = tf.multiply(2.0, x)  # Todo x in (-5.0, 10.0) or (-2.048, 2.048)
    d = x.shape[0]

    f = 0
    for i in range(d - 1):
        f += (100.0 * ((x[i + 1] - (x[i] ** 2)) ** 2)) + ((x[i] - 1) ** 2)

    return f


# f(x)=0 at 111
# @tf.function
# def lognorm_rosenbrock(x, number_free_parameters):
#     my_max = tf.math.log(rosenbrock(x=-tf.ones_like(input=x)))
#     value = my_max - tf.math.log(rosenbrock(x))  # 5 - ...
#     value /= my_max  # 20
#     value = tf.clip_by_value(value, clip_value_min=-1.0, clip_value_max=1.0)
#     return value


# f(x)=0 at 000
@tf.function
def sphere(x):
    x *= 5.0
    x = tf.pow(x, 2)
    return tf.math.reduce_sum(x, axis=0)


# f(x)=-39.17*d at (-2.9, ..., -2.9)
@tf.function
def styblinski_tang(x):
    x *= 5.0
    d = x.shape[0]

    my_sum = 0.0
    for i in range(d):
        my_sum += (x[i] ** 4) - (16 * (x[i] ** 2)) + (5.0 * x[i])

    f = 0.5 * my_sum

    return f


# f(x)=0 bei 000
@tf.function
def zakharov(x):
    # x = (((x + 1.0) / 2.0) * 15.0) - 5.0
    x *= 5
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
def normalize_function(x, number_free_parameters, objective_function, function_name):
    my_max = objective_function(x=-tf.ones_like(input=x))
    f = objective_function(x)
    d = x.shape[0]

    if function_name == "Ackley":
        my_min = objective_function(x=tf.zeros_like(input=x))
    elif function_name == "Griewank":
        my_min = objective_function(x=tf.zeros_like(input=x))
    elif function_name == "Levy":
        my_min = objective_function(x=tf.divide(tf.ones_like(input=x), 10.0))
    elif function_name == "Rastrigin":
        my_min = objective_function(x=tf.zeros_like(input=x))
    elif function_name == "Rosenbrock":
        my_min = objective_function(x=tf.divide(tf.ones_like(input=x), 2.0))
    elif function_name == "Sphere":
        my_min = objective_function(x=tf.zeros_like(input=x))
    elif function_name == "Styblinski_tang":
        my_min = -39.16599 * d
        my_min = objective_function(x=tf.divide(tf.multiply(tf.ones_like(input=x), -2.903534), 5.0))
    elif function_name == "Zakharov":
        my_min = objective_function(x=tf.zeros_like(input=x))

    value = (my_max - f) / (my_max - my_min)
    return tf.clip_by_value(value, clip_value_min=0.0, clip_value_max=1.0)


FUNCTIONS = {
    "Ackley": ackley,
    "Griewank": griewank,
    # "Langermann": langermann,
    "Levy": levy,
    # "Michalewicz": michalewicz,
    "Rastrigin": rastrigin,
    "Rosenbrock": rosenbrock,
    "Sphere": sphere,
    "Styblinski_tang": styblinski_tang,
    "Zakharov": zakharov
}
