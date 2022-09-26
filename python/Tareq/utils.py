import gin
from numpy import savetxt, loadtxt, array, float32
import os


@gin.configurable(denylist=['csv_filename', 'data_array'])
def save_array_as_csv(csv_filename, data_array, delimiter):
    savetxt(csv_filename, data_array, delimiter=delimiter)


@gin.configurable(denylist=['csv_filename'])
def load_array_from_csv(csv_filename, delimiter):
    return loadtxt(csv_filename, delimiter=delimiter)


def save_data(data, csv_dir):
    for key in data:
        save_array_as_csv(
            csv_filename=key,
            data_array=data[key]
        )
    # f = open(os.path.join(csv_dir, "summary.txt"), 'w')
    # f.writelines(summary)


def map_interval(x, a, b, c, d):
    return ((x - a) * ((d - c) / (b - a))) + c


def denormalize_x(x, a, b, c, d):
    not_normalized_state = []
    for entry in x:
        not_normalized_state.append(map_interval(entry, a, b, c, d))
    return array(not_normalized_state, dtype=float32)
