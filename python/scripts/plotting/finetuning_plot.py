import os.path

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d

from definitons import ROOT_DIR
from objective_functions.tf_objective_functions import FUNCTIONS

FINETUNING_DIR = os.path.join(ROOT_DIR, "finetuning")
base_function = [0.77, 0.91, 0.69, 0.80, 0.75, 0.78, 0.15, 0.72]
base_overall = np.mean(base_function)


def main():
    finetune_data = get_data("10_obs_0.9_gamma_100_episodes_1.0_trans_abs_env", 50000)
    baseline_data = get_data("10_obs_1.0_trans_abs_env", 0)
    data = [d[d[:, 0].argsort()] for d in baseline_data]
    data = [np.column_stack((d[:, 0], finetune_data[i][:, 1:], d[:, 1:])) for i, d in enumerate(data)]
    plot(data)
    plt.savefig(os.path.join(FINETUNING_DIR, "perf_over_time"), dpi=400, transparent=True)
    plt.show()


def plot(data):
    fig, axs = plt.subplots(2, 4)
    fig.set_size_inches(10, 6)
    h = []
    for i, (ax, function_name) in enumerate(zip(axs.flatten(), FUNCTIONS.keys())):
        y = np.array([gaussian_filter1d(a, sigma=1.00) for a in data[i][:, 1:].T]).T
        h = ax.plot(data[i][:, 0], y)
        ax.title.set_text(function_name.capitalize())
        ax.set_ylim(0, 1)
        if i != 0 and i != 4:
            ax.set_yticklabels([])

        if i < 4:
            ax.set_xticklabels([])
        ax.grid()

    plt.figlegend(labels=("Finetune function", "Finetune overall", "New function", "New overall",),
                  handles=h, loc='lower center',
                  fancybox=True, shadow=True, ncol=4)


def get_data(run_id, finetune):
    all_means = []
    for function_name in FUNCTIONS.keys():

        run_dir = os.path.join(FINETUNING_DIR,
                               "reinforce",
                               function_name,
                               run_id)
        step_dirs = [dir for dir in os.listdir(run_dir) if dir.startswith("Step_")]
        all_performances = []
        for step_dir in step_dirs:
            step = eval(step_dir.split("_")[-1])
            if finetune:
                step -= 50000
            if step > 4000:
                continue
            summary = os.path.join(run_dir, step_dir, "summary.txt")
            with open(summary) as reader:
                lines = reader.readlines()
            lines = lines[1:]
            overall_performance = np.mean([eval(line.split("=")[-1]) for line in lines])
            fct_performance = [eval(line.split("=")[-1]) for line in lines if line.startswith(function_name)][0]
            all_performances.append((step, fct_performance, overall_performance))
        all_means.append(all_performances)

    if finetune:
        data = [[(0, base_function[i], base_overall)] + mean for i, mean in enumerate(all_means)]
    else:
        data = [[(0, 0, 0)] + mean for i, mean in enumerate(all_means)]

    data = [np.array(d) for d in data]
    return data


if __name__ == "__main__":
    main()
