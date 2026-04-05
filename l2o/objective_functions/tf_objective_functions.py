import math
import tensorflow as tf
import numpy as np

# All functions must take inputs of range (-1, 1) of dimension (2, N)
# Reward normalization: (max - f) / (max - min)

dtype = tf.float32

@tf.function
def ackley(x):
    x = tf.multiply(x, 20.0)
    a, b, c = 20.0, 0.2, 2.0 * math.pi
    d = x.shape[0]
    sum1 = tf.reduce_sum(tf.pow(x, 2), axis=0)
    sum2 = tf.reduce_sum(tf.cos(c * x), axis=0)
    f = (-a * tf.exp(-b * tf.sqrt((1.0 / d) * sum1))) - tf.exp((1.0 / d) * sum2) + tf.exp(1.0) + a
    return f

@tf.function
def griewank(x):
    x = tf.multiply(x, 600.0)
    d = x.shape[0]
    my_sum = tf.reduce_sum(tf.pow(x, 2.0) / 4000.0, axis=0)
    indices = tf.range(1, d + 1, dtype=tf.float32)
    # Reshape indices for broadcasting over the batch dimension
    indices = tf.reshape(indices, (-1, 1))
    prod = tf.reduce_prod(tf.cos(x / tf.math.sqrt(indices)), axis=0)
    f = my_sum - prod + 1.0
    return f

@tf.function
def levy(x):
    x = tf.multiply(x, 10.0)
    d = x.shape[0]
    w_x = 1.0 + ((x - 1.0) / 4.0)
    term1 = tf.pow(tf.sin(math.pi * w_x[0]), 2)
    term3 = tf.pow(w_x[d-1] - 1.0, 2) * (1.0 + tf.pow(tf.sin(2.0 * math.pi * w_x[d-1]), 2))
    my_sum = tf.reduce_sum(tf.pow(w_x[:d-1] - 1.0, 2) * (1.0 + 10.0 * tf.pow(tf.sin(math.pi * w_x[:d-1] + 1.0), 2)), axis=0)
    f = term1 + my_sum + term3
    return f

@tf.function
def rastrigin(x):
    x = tf.multiply(x, 5.0)
    d, a = x.shape[0], 10.0
    my_sum = tf.reduce_sum(tf.pow(x, 2) - (a * tf.cos(2.0 * math.pi * x)), axis=0)
    f = a * d + my_sum
    return f

@tf.function
def rosenbrock(x):
    x = tf.multiply(x, 2.0)
    d = x.shape[0]
    f = tf.reduce_sum(100.0 * tf.pow(x[1:] - tf.pow(x[:-1], 2), 2) + tf.pow(x[:-1] - 1.0, 2), axis=0)
    return f

@tf.function
def sphere(x):
    x = tf.multiply(x, 5.0)
    f = tf.reduce_sum(tf.pow(x, 2), axis=0)
    return f

@tf.function
def styblinski_tang(x):
    x = tf.multiply(x, 5.0)
    my_sum = tf.reduce_sum(tf.pow(x, 4) - 16.0 * tf.pow(x, 2) + 5.0 * x, axis=0)
    f = 0.5 * my_sum
    return f

@tf.function
def zakharov(x):
    x = tf.multiply(x, 5.0)
    d = x.shape[0]
    sum1 = tf.reduce_sum(tf.pow(x, 2), axis=0)
    indices = tf.range(1, d + 1, dtype=tf.float32)
    indices = tf.reshape(indices, (-1, 1))
    sum2 = tf.reduce_sum(0.5 * indices * x, axis=0)
    f = sum1 + tf.pow(sum2, 2) + tf.pow(sum2, 4)
    return f

# Function metadata for normalization
# optimal_opt_position: the value of the optimization parameters at the global minimum
FUNCTION_METADATA = {
    "Ackley": {"opt_min": 0.0},
    "Griewank": {"opt_min": 0.0},
    "Levy": {"opt_min": 0.1},
    "Rastrigin": {"opt_min": 0.0},
    "Rosenbrock": {"opt_min": 0.5},
    "Sphere": {"opt_min": 0.0},
    "Styblinski_tang": {"opt_min": -2.903534 / 5.0},
    "Zakharov": {"opt_min": 0.0}
}

def normalize_function(x, number_free_parameters, free_values, objective_function, function_name):
    """Normalizes the objective function value to [0, 1] reward range."""
    f = objective_function(x)
    d, batch_size = x.shape[0], x.shape[1]
    
    # Calculate maximum (at the boundary of search space)
    rest_ones = tf.ones(shape=(d - number_free_parameters, batch_size), dtype=dtype)
    max_input_x = tf.concat([free_values, -1.0 * rest_ones], axis=0)
    my_max = objective_function(max_input_x)

    # Calculate minimum (at the known global minimum position)
    opt_min_val = FUNCTION_METADATA.get(function_name, {"opt_min": 0.0})["opt_min"]
    min_input_opt = tf.fill((d - number_free_parameters, batch_size), opt_min_val)
    min_input_x = tf.concat([free_values, min_input_opt], axis=0)
    my_min = objective_function(min_input_x)

    # Standard normalization to [0, 1] range
    value = (my_max - f) / (my_max - my_min)
    return tf.clip_by_value(value, 0.0, 1.0)

FUNCTIONS = {
    "Ackley": ackley,
    "Griewank": griewank,
    "Levy": levy,
    "Rastrigin": rastrigin,
    "Rosenbrock": rosenbrock,
    "Sphere": sphere,
    "Styblinski_tang": styblinski_tang,
    "Zakharov": zakharov
}
