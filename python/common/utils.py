import os

import tensorflow as tf
from tf_agents.drivers.dynamic_episode_driver import DynamicEpisodeDriver

from objective_functions.tf_objective_functions import FUNCTIONS
from definitions import ROOT_DIR, RUNS_DIR


def is_valid_filename(parser, arg):
    dir = os.path.join(ROOT_DIR, arg)
    if not os.path.exists(dir):
        parser.error("The file %s does not exist!" % arg)
    else:
        return dir


def get_run_identifiers_from_folder(folder):
    relpath = os.path.relpath(folder, RUNS_DIR)
    splitpath = relpath.split(os.path.sep)
    if len(splitpath) == 1:
        splitpath += [None, None]
    return splitpath
