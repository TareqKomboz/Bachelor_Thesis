import os

import tensorflow as tf
from tf_agents.drivers.dynamic_episode_driver import DynamicEpisodeDriver

from objective_functions.tf_objective_functions import FUNCTIONS
from definitons import ROOT_DIR, RUNS_DIR


def collect_episode(environment, policy, observers, number_episodes):
    driver = DynamicEpisodeDriver(
        environment,
        policy,
        observers,
        number_episodes=number_episodes)
    driver.run()


def is_valid_filename(parser, arg):
    dir = os.path.join(ROOT_DIR, arg)
    if not os.path.exists(dir):
        parser.error("The file %s does not exist!" % arg)
    else:
        return dir


def plot_function(domain, function):
    x = tf.range(domain[0, 0], domain[1, 0], 0.1)
    y = tf.range(domain[0, 1], domain[1, 1], 0.1)
    xy = tf.meshgrid(x, y)
    z = [function(x) for x in xy]
    return z


def format_function_names(function_names):
    if len(function_names) < 4:
        return ",".join(function_names)
    elif len(function_names) == len(FUNCTIONS.keys()):
        return "all"
    else:
        function_names = [name[:3] for name in function_names]
        return ",".join(function_names)


def get_functions_from_formatted_function_names(formatted_function_names):
    names = []
    for formatted_name in formatted_function_names.split(","):
        if formatted_name == "ack":
            names.append("ackley")
        elif formatted_name == "lan":
            names.append("langermann")
        elif formatted_name == "mic":
            names.append("michalewicz")
        elif formatted_name == "ras":
            names.append("rastrigin")
        elif formatted_name == "ros":
            names.append("rosenbrock")
        elif formatted_name == "squ":
            names.append("sumsquares")
        else:
            raise NotImplementedError("unknown function name encountered: " + formatted_name)
    return names


def get_run_identifiers_from_folder(folder):
    relpath = os.path.relpath(folder, RUNS_DIR)
    splitpath = relpath.split(os.path.sep)
    if len(splitpath) == 1:
        splitpath += [None, None]
    return splitpath
