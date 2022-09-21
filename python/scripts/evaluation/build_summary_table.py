import os
import shutil

from common.utils import get_functions_from_formatted_function_name
from objective_functions.tf_objective_functions import FUNCTIONS
from db.postgres import DBEngine, initialize_db
from definitons import RUNS_DIR
import pandas as pd

START_COLUMNS = ['algorithm', 'trained_on', 'run_id']


def scan_summary_txt(file, value_name):
    performances = []
    with open(file) as reader:
        for line in reader.readlines()[1:]:
            line = line.split(value_name)[1]
            line = line.split(",")[0]
            line = line.strip("=")
            performances.append(float(line))
    return performances


def scan_evaluation_folder(folder, value_name):
    performances = []
    for function_name in FUNCTIONS.keys():
        performances += scan_summary_txt(os.path.join(folder, function_name, "summary.txt"), value_name)
    return performances


def has_translation_in_training(run_id_dir):
    configfile = os.path.join(run_id_dir, 'config.gin')
    with open(configfile) as reader:
        lines = reader.readlines()
        line = list(filter(lambda line: line.startswith('train.translation'), lines))[0]
    return eval(line.strip('\n').split('=')[1]) > 0.0


def delete_if_empty(full_run_id_path, step_dirs):
    evaluation_folder = os.path.join(full_run_id_path, step_dirs[0])
    try:
        assert os.path.isfile(os.path.join(evaluation_folder, "summary.txt"))
    except AssertionError as e:
        print("no summary found here: " + evaluation_folder)
        if len(step_dirs) == 1:
            print("The only step dir is empty deleting run folder")
            shutil.rmtree(full_run_id_path)
            return True
        else:
            shutil.rmtree(evaluation_folder)
            print("Faulty step dir configuration, latest step dir is deleted, please rerun the script")
            raise BaseException
    return False


def find_evaluation_folders():
    alg_dirs = os.listdir(RUNS_DIR)
    alg_dirs = [x for x in alg_dirs if os.path.isdir(os.path.join(RUNS_DIR, x))]
    algs = []
    trained_ons = []
    run_ids = []
    evaluation_folders = []
    translations = []
    for alg_dir in alg_dirs:
        full_alg_path = os.path.join(RUNS_DIR, alg_dir)
        if is_not_learned(alg_dir):
            evaluation_folders.append(full_alg_path)
            try:
                assert os.path.isfile(os.path.join(evaluation_folders[-1], "summary.txt"))
            except AssertionError as e:
                print("no summary found here: " + full_alg_path)
                raise e
            algs.append(alg_dir)
            trained_ons.append(None)
            run_ids.append(None)
            translations.append(False)
            continue
        trained_on_dirs = os.listdir(full_alg_path)
        for trained_on_dir in trained_on_dirs:
            full_trained_on_path = os.path.join(full_alg_path, trained_on_dir)
            run_id_dirs = os.listdir(full_trained_on_path)
            for run_id_dir in run_id_dirs:
                translations.append(has_translation_in_training(os.path.join(full_trained_on_path, run_id_dir)))
                full_run_id_path = os.path.join(full_trained_on_path, run_id_dir)
                step_dirs = os.listdir(full_run_id_path)
                step_dirs = [x for x in step_dirs if os.path.isdir(os.path.join(full_run_id_path, x))]
                step_dirs = [x for x in step_dirs if not x.startswith("checkpoint")]
                step_dirs = [x for x in step_dirs if not x.startswith("_")]
                step_dirs.sort(key=lambda x: int(x.split("Step_")[1]), reverse=True)
                if len(step_dirs) == 0:
                    print("No Step Dirs - " + full_run_id_path + " deleting folder")
                    shutil.rmtree(full_run_id_path)
                    continue
                evaluation_folder = os.path.join(full_run_id_path, step_dirs[0])
                if delete_if_empty(full_run_id_path, step_dirs):
                    continue
                algs.append(alg_dir)
                trained_ons.append(trained_on_dir)
                run_ids.append(run_id_dir)
                evaluation_folders.append(evaluation_folder)

    return evaluation_folders, algs, trained_ons, run_ids, translations


def is_not_learned(alg_dir):
    return False


def find_summaries_and_write_to_file(file, value_name="train_final"):
    print("Creating summary table")
    columns = START_COLUMNS
    for function_name in FUNCTIONS.keys():
        columns.append("{}".format(function_name))

    evaluation_folders, algorithms, trained_ons, run_ids, translations = find_evaluation_folders()
    print("Found {} runs to summarize".format(len(evaluation_folders)))
    data = []
    for folder in evaluation_folders:
        data.append(scan_evaluation_folder(folder, value_name))
    data = [[algorithms[i]] + [trained_ons[i]] + [run_ids[i]] + data[i] for i in range(len(data))]
    df = pd.DataFrame(data=data, columns=columns)
    df.loc[(df.trained_on == "ackley,griewank,levy,rastrigin,rosenbrock,sphere,styblinski_tang,zakharov"),
           'trained_on'] = 'all'
    df['overall'] = 0
    for function_name in FUNCTIONS.keys():
        df['overall'] += df[function_name]
    df['overall'] /= len(FUNCTIONS.keys())

    df['in_distribution'] = 0
    df['out_of_distribution'] = 0
    for i, (trained_on, translation) in enumerate(zip(trained_ons, translations)):
        if trained_on is None:
            df['out_of_distribution'][i] = df['overall'][i]
            df['in_distribution'][i] = df['overall'][i]
        elif trained_on == 'all':
            for name in FUNCTIONS.keys():
                df['in_distribution'][i] += df['{}_control'.format(name)][i]
        elif len(trained_on.split(",")) >= 4:
            function_name = get_functions_from_formatted_function_name(trained_on)
            for name in function_name:
                df['in_distribution'][i] += df['{}_control'.format(name)][i]
        elif len(trained_on.split(",")) > 1:
            function_name = trained_on.split(",")
            for name in function_name:
                df['in_distribution'][i] += df['{}_control'.format(name)][i]

        else:
            df['in_distribution'][i] += df['{}_control'.format(trained_on)][i]
            for name in FUNCTIONS.keys():
                if name == trained_on:
                    continue
                df['out_of_distribution'][i] += 5 * df[name][i]

    df.to_csv(file)
    print("Summary written to: {}".format(file))


def plot_table_by_function_values(file):
    df = pd.read_csv(file)
    df_by_function = df[START_COLUMNS]
    df_by_function['overall'] = df['overall']
    for function_name in FUNCTIONS.keys():
        df_by_function[function_name] = df[function_name]

    file2 = os.path.join(RUNS_DIR, "{}.csv".format("small summary"))
    df_by_function.to_csv(file2)


def insert_into_postgres(file):
    initialize_db()
    connector = DBEngine()
    df = pd.read_csv(file)
    n_rows = df.to_sql('performance', connector.engine, if_exists='replace')
    print("inserted {} rows into table {}".format(n_rows, connector.dbname))


if __name__ == "__main__":
    file = os.path.join(RUNS_DIR, "{}.csv".format("final summary"))
    find_summaries_and_write_to_file(file, "train_final")
    insert_into_postgres(file)
