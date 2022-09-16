import os

import matplotlib.pyplot as plt
import numpy as np

import matplotlib.colors as mcolors

from definitons import ROOT_DIR
from db.runs import read_from_sql
from objective_functions.tf_objective_functions import FUNCTIONS

EPISODE_LENGTH = 50
FIG_NAME = "top_over_perf"
PLOT_BY = "operator"
type = "overall"

if type == "single":
    TOP_PERFORMERS = (("reinforce", "ackley", "10_obs_0.9_gamma_1.0_trans_abs_env", "ackley"),
                      ("reinforce", "rastrigin", "10_obs_0.5_gamma_100_episodes_1.0_trans_abs_env", "rastrigin"),
                      ("reinforce", "sphere", "10_obs_0.9_gamma_100_episodes_1.0_trans_abs_env", "sphere"),
                      ("reinforce", "rosenbrock", "10_obs_0.9_gamma_1.0_trans_abs_env", "rosenbrock"),
                      ("reinforce", "michalewicz", "10_obs_1.0_gamma_1.0_trans_abs_env", "michalewicz"))
elif type == "overall":
    TOP_PERFORMERS = (("reinforce", "all", "10_obs_0.9_gamma_100_episodes_1.0_trans_abs_env"),
                      ("reinforce", "all", "25_obs_giant_vnet_deep_pnet_rel_env"))

elif type == "single_rel":

    FIG_NAME += "_rel"
    TOP_PERFORMERS = (("reinforce", "ackley", "10_obs_rel_env", "ackley"),
                      ("reinforce", "rastrigin", "25_obs_double_pnet_rel_env", "rastrigin"),
                      ("reinforce", "sphere", "25_obs_1.0_trans_rel_env", "sphere"),
                      ("reinforce", "rosenbrock", "25_obs_1.0_gamma_rel_env", "rosenbrock"),
                      ("reinforce", "michalewicz", "25_obs_1.0_trans_rel_env", "michalewicz"))
else:
    raise NotImplementedError


OPERATORS = ("control")
PLOT_DIR = os.path.join(ROOT_DIR, "comparison_plots")
COLORS = [color for color in mcolors.TABLEAU_COLORS.values()]


def main():
    if PLOT_BY == "run_on" and type.startswith("single"):
        means, stds = get_data_by_run_on()
        plot_by_run_on(means, stds, ("Single", "All"))
    elif PLOT_BY == "run_on" and type.startswith("overall"):
        means, stds = get_data_by_run_on_overall()
        plot_by_run_on(means, stds, ("Abs All", "Rel All"))
    elif PLOT_BY == "operator":
        means, stds = get_data_by_operator()
        plot_by_operator(means, stds)
    elif PLOT_BY == "operator_over_time":
        means, stds = get_data_by_operator_over_time()
        plot_by_run_on(means, stds, [a.capitalize().replace("_", " ") for a in OPERATORS])
    else:
        raise NotImplementedError
    plt.savefig(os.path.join(PLOT_DIR, "{}_{}_{}".format(FIG_NAME, EPISODE_LENGTH, PLOT_BY))
                , dpi=400, transparent=True)
    plt.show()


def plot_by_operator(all_means, all_stds):
    fig, axss = plt.subplots(8, 4)
    fig.set_size_inches(10, 15)
    plt.subplots_adjust(right=0.95, left=0.1, bottom=0.05, top=0.95, wspace=0.1, hspace=0.15)
    labels = ("Absolute All", "Relative All")

    for j, (axs, function_name) in enumerate(zip(axss, FUNCTIONS.keys())):
        for i, (ax, label) in enumerate(zip(axs, labels)):
            if j == 0:
                ax.title.set_text(label)
                ax.title.set_size(16)
            if i == 0:
                ax.text(-0.25, 0.5, function_name.capitalize(),
                        horizontalalignment='right',
                        verticalalignment='center',
                        rotation='vertical',
                        transform=ax.transAxes, fontdict={'size': 16})
            else:
                ax.set_yticklabels([])
            if j < 7:
                ax.set_xticklabels([])
            means = all_means[i][j]
            stds = all_stds[i][j]
            h = plot_performance_over_time_with_stds(ax, range(EPISODE_LENGTH), means, stds, std_scale=0.25)
            ax.set_ylim(0, 1)
            ax.grid(axis='y')

    plt.figlegend(labels=[op.capitalize().replace("_", " ") for op in OPERATORS], handles=h, loc='lower center',
                  fancybox=True, shadow=True, ncol=5)


def plot_by_run_on(all_means, all_stds, labels):
    fig, axs = plt.subplots(2, 4)
    fig.set_size_inches(10, 6)

    for i, (ax, means) in enumerate(zip(axs.flatten(), all_means)):
        stds = all_stds[i]
        means = [mean[:EPISODE_LENGTH] for mean in means]
        stds = [std[:EPISODE_LENGTH] for std in stds]
        h = plot_performance_over_time_with_stds(ax,
                                                 range(EPISODE_LENGTH),
                                                 means,
                                                 stds,
                                                 std_scale=0.25)

        ax.title.set_text([n for n in FUNCTIONS.keys()][i].capitalize())
        ax.set_ylim(0, 1)
        if i != 0 and i != 4:
            ax.set_yticklabels([])

        if i < 4:
            ax.set_xticklabels([])
        ax.grid(axis='y')
    plt.figlegend(labels=labels, handles=h,
                  loc="lower center", fancybox=True, shadow=True, ncol=6)


def plot_performance_over_time_with_stds(ax, x, means, stds, std_scale=0.1):
    h = []
    for i, (mean, std) in enumerate(zip(means, stds)):
        h += ax.plot(x, mean, color=COLORS[i])
        ax.fill_between(x, mean, mean - std_scale * std, color=COLORS[i], alpha=0.2)
        ax.fill_between(x, mean, mean + std_scale * std, color=COLORS[i], alpha=0.2)
    return h


def plot_performance_by_labels(ax, labels, performance, width=0.5):
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=25)
    return ax.bar(labels, performance, width, color=mcolors.TABLEAU_COLORS)


def get_data_by_operator():
    all_means = []
    all_stds = []
    for top_performer in TOP_PERFORMERS:
        df = read_from_sql(False,
                           (top_performer[0],),
                           (top_performer[1],),
                           (top_performer[2],),
                           "all",
                           "all")
        means = []
        stds = []
        for function_name in FUNCTIONS.keys():
            means.append(np.array(
                [convert_series_to_array(
                    df[(df['operation'] == operator) & (df['run_on'] == function_name)]['mean_perf_over_time'])[0][:EPISODE_LENGTH]
                 for operator in OPERATORS]
            ))
            stds.append(np.array(
                [convert_series_to_array(
                    df[(df['operation'] == operator) & (df['run_on'] == function_name)]['std_perf_over_time'])[0][:EPISODE_LENGTH]
                 for operator in OPERATORS]
            ))
        all_means.append(means)
        all_stds.append(stds)
    return np.array(all_means), np.array(all_stds)


def get_data_by_operator_over_time():
    all_means = []
    all_stds = []
    for top_performer in TOP_PERFORMERS:
        df = read_from_sql(False,
                           (top_performer[0],),
                           (top_performer[1],),
                           (top_performer[2],),
                           "all",
                           (top_performer[1],))
        means = np.array(
            [convert_series_to_array(df[df['operation'] == operator]['mean_perf_over_time'])[0]
             for operator in OPERATORS]
        )
        stds = np.array(
            [convert_series_to_array(df[df['operation'] == operator]['std_perf_over_time'])[0]
             for operator in OPERATORS]
        )
        all_means.append(means)
        all_stds.append(stds)
    return all_means, all_stds


def get_data_by_run_on_overall():
    all_means = []
    all_stds = []
    for i, function_name in enumerate(FUNCTIONS.keys()):
        means = []
        stds = []
        '''
        '''
        baseline_df = read_from_sql(False,
                                    ["random_search", "powell"],
                                    [""],
                                    [""],
                                    "all",
                                    (function_name,))

        group_by = baseline_df.groupby("algorithm")
        for group in group_by:
            means.append(np.mean(convert_series_to_array(group[1]['mean_perf_over_time']), axis=0))
            stds.append(np.mean(convert_series_to_array(group[1]['std_perf_over_time']), axis=0))

        for top_performer in TOP_PERFORMERS:
            single_df = read_from_sql(False,
                                      (top_performer[0],),
                                      (top_performer[1],),
                                      (top_performer[2],),
                                      "all",
                                      (function_name,))
            means.append(np.mean(convert_series_to_array(single_df['mean_perf_over_time']), axis=0))
            stds.append(np.mean(convert_series_to_array(single_df['std_perf_over_time']), axis=0))

        all_means.append(means)
        all_stds.append(stds)
    return all_means, all_stds


def get_data_by_run_on():
    all_means = []
    all_stds = []
    for top_performer in TOP_PERFORMERS:
        single_df = read_from_sql(False,
                                  (top_performer[0],),
                                  (top_performer[1],),
                                  (top_performer[2],),
                                  "all",
                                  (top_performer[1],))

        baseline_df = read_from_sql(True,
                                    [],
                                    [],
                                    [],
                                    "all",
                                    (top_performer[1],))
        means = []
        stds = []
        group_by = baseline_df.groupby("algorithm")
        for group in group_by:
            means.append(np.mean(convert_series_to_array(group[1]['mean_perf_over_time']), axis=0))
            stds.append(np.mean(convert_series_to_array(group[1]['std_perf_over_time']), axis=0))

        means.append(np.mean(convert_series_to_array(single_df['mean_perf_over_time']), axis=0))
        stds.append(np.mean(convert_series_to_array(single_df['std_perf_over_time']), axis=0))

        all_means.append(means)
        all_stds.append(stds)

    return all_means, all_stds


def convert_series_to_array(series):
    return [np.array(element.strip("{").strip("}").split(","), np.float64)[:EPISODE_LENGTH] for element in series]


if __name__ == "__main__":
    main()
