import argparse
import os
import subprocess
import sys
import time

from l2o.objective_functions.tf_objective_functions import FUNCTIONS
from l2o.common.utils import is_valid_filename
from l2o.definitions import ROOT_DIR


def change_if_function_name(line, name):
    if line.startswith('main.function_name'):
        line = line.split("=")
        line[1] = name
        line = line[0] + "= " + line[1]
    return line


MAIN_SCRIPT = os.path.join(ROOT_DIR, "main.py")


def create_configfiles(configfile):
    function_name = ["(\"{}\",)".format(name) for name in FUNCTIONS.keys()] + ["\"all\""]
    new_configfiles = [configfile.strip(".gin") + "_" + name + ".gin" for name in FUNCTIONS.keys()] +\
                      [configfile.strip(".gin") + "_all.gin"]
    for i, name in enumerate(function_name):
        with open(configfile) as reader:
            lines = reader.readlines()
            lines = [change_if_function_name(line, name) for line in lines]
        with open(new_configfiles[i], 'w') as writer:
            writer.writelines(lines)
    return new_configfiles


def main(arguments):
    main_script = os.path.join(PYTHON_DIR, "main.py")
    configfiles = create_configfiles(arguments.configfile)
    start = time.time()
    processes = []
    for i, configfile in enumerate(configfiles):
        command = "{} {} -c {}".format(sys.executable, main_script, configfile)
        print("{}: Starting Training with configfile: {}".format(i, os.path.relpath(configfile, ROOT_DIR)))
        print("{}: running command: {}".format(i, command))
        processes.append(subprocess.Popen(command, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT))

    while True:
        timestamp = time.strftime("%H:%M:%S", time.gmtime(time.time() - start))
        stati = []
        for p in processes:
            stati.append(p.poll())
        if any([status is None for status in stati]):
            print(end="\r")
            print("processes running, {}".format(timestamp), end="")
            continue
        elif all([status == 0 for status in stati]):
            print(end="\r")
            print("batch finished, {}".format(timestamp))
            break
        else:
            print(end="\r")
            print("some command failed, all stati: {}, {}".format(stati, timestamp))
            break

    timestamp = time.strftime("%H:%M:%S", time.gmtime(time.time() - start))
    print("all finished, {}".format(timestamp))


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser(description="gets gin config file")
    arg_parser.add_argument("-c",
                            "--config",
                            dest="configfile",
                            help="provide gin config file path relative to root directory",
                            type=lambda x: is_valid_filename(arg_parser, x),
                            default="default.gin")
    args = arg_parser.parse_args()
    main(arguments=args)
