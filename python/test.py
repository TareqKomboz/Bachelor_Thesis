import tensorflow as tf
import numpy as np


def sphere(x):
    x = tf.multiply(x, 5.0)
    d = x.shape[0]

    my_sum = 0.0
    for i in range(d):
        my_sum += tf.pow(x[i], 2)

    return my_sum


def build_evaluation_parameters(n_start_pos, input_dimension):
    N_start_pos = n_start_pos ** input_dimension

    my_list = []
    my_linspace = np.linspace(-0.9, 0.9, num=n_start_pos, dtype=np.float32)
    for i in range(input_dimension):
        my_list.append(my_linspace)

    mesh = np.meshgrid(*tuple(my_list), sparse=True)
    transposed_mesh = np.transpose(mesh)
    reshaped_mesh = np.reshape(transposed_mesh, (N_start_pos, input_dimension))

    return reshaped_mesh


if __name__ == "__main__":
    mesh = build_evaluation_parameters(4, 10)
    sphere(mesh)
