import glob
import os

from common.build_run_id import get_run_id
from definitons import RUNS_DIR

CONFIGFILES = glob.glob(os.path.join(RUNS_DIR, '*/*/*/config.gin'))

def parse_value(line):
    return eval(line.split("=")[1].strip(" "))

def change_line_if_run_id(line, run_id):
    if line.startswith("main.run_id"):
        old_run_id = parse_value(line)
        line = line.replace(old_run_id, run_id)
    return line


def rename_run_id(configfile, run_id):
    with open(configfile) as reader:
        lines = reader.readlines()
    new_lines = [change_line_if_run_id(line, run_id) for line in lines]
    with open(configfile, 'w') as writer:
        writer.writelines(new_lines)
    evaluation_folder = os.path.split(configfile)[0]
    new_evaluation_folder = os.path.join(os.path.split(evaluation_folder)[0], run_id)
    os.rename(evaluation_folder, new_evaluation_folder)
    print("renamed {} to {}".format(evaluation_folder, new_evaluation_folder))


def main():
    for configfile in CONFIGFILES:
        run_id = get_run_id(configfile)
        rename_run_id(configfile, run_id)


if __name__ == "__main__":
    print("{} configfiles found, building their unified run-ids".format(len(CONFIGFILES)))
    main()
