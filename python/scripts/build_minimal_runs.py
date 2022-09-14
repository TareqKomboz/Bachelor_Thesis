import os
import shutil
from pathlib import Path

from definitons import RUNS_DIR, ROOT_DIR

MIN_RUNS_DIR = os.path.join("D:", "runs")

def main():
    alg_dirs = os.listdir(RUNS_DIR)
    alg_dirs = [x for x in alg_dirs if os.path.isdir(os.path.join(RUNS_DIR, x))]
    for alg_dir in alg_dirs:
        if is_baseline(alg_dir):
            continue
        full_alg_dir = os.path.join(RUNS_DIR, alg_dir)
        fct_dirs = os.listdir(full_alg_dir)
        for fct_dir in fct_dirs:
            full_fct_dir = os.path.join(full_alg_dir, fct_dir)
            run_dirs = os.listdir(full_fct_dir)
            for run_dir in run_dirs:
                full_run_dir = os.path.join(full_fct_dir, run_dir)
                rel_run_dir = os.path.relpath(full_run_dir, RUNS_DIR)
                full_checkpoint_dir = os.path.join(full_run_dir, "checkpoint")
                files = [
                    os.path.join(full_run_dir, "config.gin"),
                    os.path.join(full_run_dir, "eval.log"),
                    os.path.join(full_run_dir, "run.log"),
                    os.path.join(full_run_dir, "train_returns.csv"),
                    os.path.join(full_run_dir, "train_performance.csv"),
                    os.path.join(full_run_dir, "train_losses.csv"),
                         ]
                for file in files:
                    try:
                        assert os.path.isfile(file)
                    except AssertionError as e:
                        print("file does not exist: {}".format(file))
                        if file.endswith("summary.txt"):
                            raise AssertionError("config is needed")
                        files = [good_file for good_file in files if good_file != file]
                print("copying {} to minimal runs folder".format(rel_run_dir))
                copy(files, full_checkpoint_dir, os.path.join(MIN_RUNS_DIR, rel_run_dir))


def copy(files, checkpoint_dir, destination):
    Path(destination).mkdir(parents=True, exist_ok=True)
    shutil.copytree(checkpoint_dir, os.path.join(destination, 'checkpoint'), dirs_exist_ok=True)
    for file in files:
        shutil.copy2(file, destination)

def is_baseline(alg_dir):
    return alg_dir.endswith("powell") or alg_dir.endswith("nelder-mead") \
           or alg_dir.endswith("random_search") or alg_dir.endswith("grid_search") \
           or alg_dir.endswith("random_search_200")


if __name__ == "__main__":
    main()