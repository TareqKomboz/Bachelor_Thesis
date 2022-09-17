import glob
import os


from definitons import RUNS_DIR


def change(starting_pattern, change_fn):
    configfiles = glob.glob(os.path.join(RUNS_DIR, '*/*/*/config.gin'))
    for file in configfiles:
        changed = False
        newlines = []
        with open(file, 'r') as reader:
            lines = reader.readlines()
            for line in lines:
                if line.startswith(starting_pattern):
                    line = change_fn(line)
                    changed = True
                newlines.append(line)
        if changed:
            with open(file, 'w', encoding='utf8') as writer:
                writer.writelines(newlines)
            print('{} changed'.format(file))


def multiple_functions(line):
    line = line.replace("main.function_name ", "main.function_names")
    function_name = line.split(" ")[-1].strip("\n")
    return line.replace(function_name, "({},)".format(function_name))


def change_run_id(line):
    return line.replace("no_vnet", "no_rand_trans")


def add_operators(line):
    if eval(line.split("=")[1]):
        return line.replace(line)
    else:
        return line.replace(line)


if __name__ == "__main__":
    change("train.randomize_translation", add_operators)
