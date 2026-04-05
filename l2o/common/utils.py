import os
import tensorflow as tf
from numpy import savetxt, loadtxt, array, float32, ones
from tf_agents.drivers.dynamic_episode_driver import DynamicEpisodeDriver
import gin

from l2o.definitions import ROOT_DIR, RUNS_DIR


def is_valid_filename(parser, arg):
    directory = os.path.join(ROOT_DIR, arg)
    if not os.path.exists(directory):
        parser.error("The file %s does not exist!" % arg)
    else:
        return directory


def get_run_identifiers_from_folder(folder):
    relpath = os.path.relpath(folder, RUNS_DIR)
    splitpath = relpath.split(os.path.sep)
    if len(splitpath) == 1:
        splitpath += [None, None]
    return splitpath


@gin.configurable(denylist=['csv_filename', 'data_array'])
def save_array_as_csv(csv_filename, data_array, delimiter=','):
    """Saves a numpy array to a CSV file."""
    savetxt(csv_filename, data_array, delimiter=delimiter)


@gin.configurable(denylist=['csv_filename'])
def load_array_from_csv(csv_filename, delimiter=','):
    """Loads a numpy array from a CSV file."""
    return loadtxt(csv_filename, delimiter=delimiter)


def save_data(data, csv_dir=None):
    """Saves a dictionary of {filename: array} to CSV files."""
    for key in data:
        save_array_as_csv(
            csv_filename=key,
            data_array=data[key]
        )


def map_interval(x, a, b, c, d):
    """Maps a value from interval [a, b] to [c, d]."""
    return ((x - a) * ((d - c) / (b - a))) + c


def denormalize_x(x, a, b, c, d):
    """Denormalizes a state array from [a, b] back to [c, d]."""
    not_normalized_state = []
    for entry in x:
        not_normalized_state.append(map_interval(entry, a, b, c, d))
    return array(not_normalized_state, dtype=float32)
