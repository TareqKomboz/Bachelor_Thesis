import argparse
import os
import glob
import subprocess
import sys
import time

import gin

from common.utils import get_run_identifiers_from_folder
from definitions import PYTHON_DIR, RUNS_DIR, ROOT_DIR
from main import main


# loads default evaluation parameters from default.gin
# then finds all config.gin in the runs folder and runs an evaluation for every single one, with 8 parallel threads
# if run_all is not set, runs in completed.csv will be ignored

MAIN_SCRIPT = os.path.join(PYTHON_DIR, "main.py")
CONFIGFILES = glob.glob(os.path.join(RUNS_DIR, '*/*/*/config.gin'))
COMPLETED_FILE = os.path.join(RUNS_DIR, "completed.csv")


class Args:
    def __init__(self, configfile, debug, evaluate):
        self.evaluate = evaluate
        self.configfile = configfile
        self.debug = debug


def filter_configfiles(args):
    with open(COMPLETED_FILE) as reader:
        completed_identifiers = set([line.strip('\n') for line in reader.readlines()])

    configfiles_to_run = CONFIGFILES if args.run_all else []

    for i, file in enumerate(CONFIGFILES):
        identifiers = get_run_identifiers_from_folder(os.path.split(file)[0])
        identifier_string = ",".join(identifiers)
        if not args.run_all and identifier_string not in completed_identifiers:
            configfiles_to_run.append(file)
            print("{}: {}".format(i, os.path.relpath(file, RUNS_DIR)))
            completed_identifiers.add(identifier_string)
        elif args.run_all:
            print("{}: {}".format(i, os.path.relpath(file, RUNS_DIR)))

    if len(configfiles_to_run) == 0:
        print("all evaluations are up to date, run with option [--run_all] to force re-runs")
    else:
        print("Found {} configfiles in {}, evaluating all:".format(len(configfiles_to_run), RUNS_DIR))
        run_evaluations(configfiles_to_run, args.threads)

        if args.run_all:
            completed_identifiers = [",".join(get_run_identifiers_from_folder(os.path.split(file)[0])
                                              for file in CONFIGFILES)]

        with open(COMPLETED_FILE, 'w') as writer:
            writer.writelines([line + "\n" for line in completed_identifiers])


def run_evaluations(configfiles, n):
    start = time.time()
    i = 0
    processes = []
    split_configfiles = split(configfiles, n)
    for configfiles in split_configfiles:
        timestamp = time.strftime("%H:%M:%S", time.gmtime(time.time() - start))
        print("starting batch with {} configfiles, {}".format(len(configfiles), timestamp))
        for configfile in configfiles:
            command = "{} {} -e True -c {}".format(sys.executable, MAIN_SCRIPT, configfile)
            print("{}: Next Evaluation with configfile: {}".format(i, os.path.relpath(configfile, RUNS_DIR)))
            # print("{}: running command: {}".format(i, command))
            processes.append(subprocess.Popen(command, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT))
            i += 1
        while True:
            time.sleep(1)
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


def split(arr, size):
    arrs = []
    while len(arr) > size:
        piece = arr[:size]
        arrs.append(piece)
        arr = arr[size:]
    arrs.append(arr)
    return arrs


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser(description="Number of parallel runs")
    arg_parser.add_argument("-t",
                            "--threads",
                            dest="threads",
                            help="provide number of parallel calls",
                            type=int,
                            default=8)
    arg_parser.add_argument("-a",
                            "--run-all",
                            dest="run_all",
                            help="if flag is set, all are run, if not, only the ones,"
                                 " that aren't in the completed file are run",
                            type=bool,
                            action=argparse.BooleanOptionalAction,
                            default=False)
    arguments = arg_parser.parse_args()
    gin.parse_config_file(os.path.join(ROOT_DIR, 'default.gin'))
    filter_configfiles(arguments)
