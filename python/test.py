import tensorflow as tf
import math


def sumsquares(x):
    x = tf.pow(x, 2)
    return -(tf.math.reduce_sum(x, axis=0))


if __name__ == "__main__":
    x = tf.random.uniform(shape=(512, 2), minval=(-1, -1), maxval=(1, 1), dtype=tf.float32)
    x_transposed = tf.transpose(x)
    print(sumsquares(x=x_transposed))


