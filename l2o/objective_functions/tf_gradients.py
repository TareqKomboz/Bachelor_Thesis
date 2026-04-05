import math
import tensorflow as tf

@tf.function
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


@tf.function
def griewank_gradient(x):
    d = x.shape[0]
    gradient = []
    for i in range(d):
        casted_i = tf.cast(i + 1, dtype=tf.float32)
        gradient.append((x[i] / 2000.0) + ((1 / tf.sqrt(casted_i)) * tf.sin(tf.math.divide(x[i], tf.math.sqrt(casted_i)))))

    return gradient


@tf.function
def levy_gradient(x):
    d = x.shape[0]
    gradient = []
    for i in range(d):
        if i == 0:
            term1 = 0.5 * math.pi * tf.sin(math.pi * w(x[i])) * tf.cos(math.pi * w(x[i]))
            term2 = 0.5 * (w(x[i]) - 1) * (1 + 10 * (tf.sin(math.pi * w(x[i]) + 1) ** 2)) + \
                    5 * math.pi * ((w(x[i]) - 1) ** 2) * tf.sin((math.pi * w(x[i])) + 1) * tf.cos((math.pi * w(x[i])) + 1)
            gradient.append(term1 + term2)
        elif i == d - 1:
            term3 = 0.5 * (w(x[i]) - 1) * (1 + (tf.sin(2 * math.pi * w(x[i])) ** 2)) + \
                    math.pi * ((w(x[i]) - 1) ** 2) * tf.sin(2 * math.pi * w(x[i])) * tf.cos(2 * math.pi * w(x[i]))
            gradient.append(term3)
        else:
            term2 = 0.5 * (w(x[i]) - 1) * (1 + 10 * (tf.sin(math.pi * w(x[i]) + 1) ** 2)) + \
                    5 * math.pi * ((w(x[i]) - 1) ** 2) * tf.sin((math.pi * w(x[i])) + 1) * tf.cos((math.pi * w(x[i])) + 1)
            gradient.append(term2)

    return gradient


@tf.function
def w(x_i):
    return 1 + ((x_i - 1) / 4)


@tf.function
def rastrigin_gradient(x):
    d = x.shape[0]
    gradient = []
    for i in range(d):
        gradient.append((2 * x[i]) + (10 * 2 * math.pi * tf.sin(2 * math.pi * x[i])))

    return gradient


@tf.function
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
def sphere_gradient(x):
    d = x.shape[0]
    gradient = []
    for i in range(d):
        gradient.append(2*x[i])

    return gradient


@tf.function
def styblinski_tang_gradient(x):
    d = x.shape[0]
    gradient = []
    for i in range(d):
        gradient.append(0.5 * ((4 * (x[i] ** 3)) - (32 * x[i]) + 5))

    return gradient


@tf.function
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


GRADIENTS = {
    "Ackley": ackley_gradient,
    "Griewank": griewank_gradient,
    # "Langermann": langermann_gradient,
    "Levy": levy_gradient,
    # "Michalewicz": michalewicz_gradient,
    "Rastrigin": rastrigin_gradient,
    "Rosenbrock": rosenbrock_gradient,
    "Sphere": sphere_gradient,
    "Styblinski_tang": styblinski_tang_gradient,
    "Zakharov": zakharov_gradient
}