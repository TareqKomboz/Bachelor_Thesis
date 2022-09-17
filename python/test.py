import tensorflow as tf
from tf_agents.utils import common
from numpy import array, ones


def understand(_steps, _num_observations, _observations):
    if tf.less(_steps, _num_observations):
        observation = tf.pad(
            _observations[:, _steps:: -1],
            [[0, 0], [0, _num_observations - _steps - 1], [0, 0]]
    )
    else:
        observation = _observations[:, _steps:_steps - _num_observations:-1]

    return observation

def no_return():
    a = 4


if __name__ == "__main__":
    # x = tf.random.uniform(shape=(512, 2), minval=(-1, -1), maxval=(1, 1), dtype=tf.float32)
    # x_transposed = tf.transpose(x)
    #
    # batch_size = 512
    # episode_length = 50
    # observation_shape =
    # d_type = tf.float32
    #
    # _steps = common.create_variable('step', 0)
    # num_observations = 2
    # _num_observations = tf.constant(num_observations, dtype=tf.int64)
    #
    # _observations = common.create_variable(
    #     'observations',
    #     shape=(batch_size, episode_length, observation_shape),
    #     dtype=_dtype
    # )
    #
    # understand(_steps=_steps, _num_observations=_num_observations, _observations=_observations)

    # a = array([
    #     [
    #         [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    #         [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    #         [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    #         [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    #         [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    #         [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    #         [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    #     ],
    #     [
    #         [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    #         [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    #         [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    #         [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    #         [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    #         [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    #         [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    #     ]
    # ])
    x = tf.random.uniform(
            shape=(16, 2),
            minval=tuple((-1 * ones((2,), dtype=int)).tolist()),
            maxval=tuple((ones((2,), dtype=int)).tolist()),
            dtype=tf.float32
        )
    x = tf.reshape(
        x,
        (1, int(16 / 1), 2)
    )

    print(len(x))


