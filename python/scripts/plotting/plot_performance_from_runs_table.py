import os

import gin
import numpy as np
import pandas as pd
import argparse
from natsort import index_natsorted, order_by_index

from matplotlib import pyplot as plt
import matplotlib.colors as mcolors

from objective_functions.tf_objective_functions import FUNCTIONS
from db.runs import read_from_sql
from definitons import ROOT_DIR

PLOT_DIR = os.path.join(ROOT_DIR, "comparison_plots")
COLORS = [color for color in mcolors.TABLEAU_COLORS.values()]


@gin.configurable
class Args:
    def __init__(self, name, algorithms, trained_ons, run_ids, run_ons,
                 episode_length, subplots, barplots, plotby, same_as_trained, labels):
        self.same_as_trained = same_as_trained
        self.plotby = plotby
        self.barplots = barplots
        self.subplots = subplots
        self.episode_length = episode_length
        self.run_ons = run_ons
        self.run_ids = run_ids
        self.algorithms = algorithms
        self.name = name
        self.trained_ons = [name for name in FUNCTIONS.keys()] if trained_ons == "all" else trained_ons
        self.labels = labels


def main(args):
    df = get_data(args)
    plt.yticks(np.arange(0, 1.1, 0.1))
    plt.xlabel("steps")
    plt.ylabel("performance")
    plt.title(args.name)
    plt.ylim(0, 1)
    plt.tight_layout()
    df = df.reindex(index=order_by_index(df.index, index_natsorted(df['algorithm'])))
    name = arguments.name
    if arguments.barplots:
        name += "-bar"
    name += ".png"
    if args.subplots:
        if args.plotby == "run_on":
            if "all" in args.run_ons:
                plot_by_run_on(args, df)
        else:
            raise NotImplementedError("Plot by {} has not been implemented yet".format(args.plotby))

    plt.savefig(os.path.join(PLOT_DIR, name), dpi=400, transparent=True)
    plt.show()


def plot_by_run_on(args, df):
    fig, axs = plt.subplots(2, 4)
    fig.set_size_inches(10, 6)
    for i, (function_name, ax) in enumerate(zip(FUNCTIONS.keys(), axs.flatten())):

        means = df[df['run_on'] == function_name]['mean_perf_over_time']
        stds = df[df['run_on'] == function_name]['std_perf_over_time']
        if args.barplots:

            plt.grid(axis='y')
            means = [mean[args.episode_length - 1] for mean in means]

            h = plot_performance_by_labels(ax, args.labels, means)
            plt.figlegend(labels=[name for name in args.labels],
                          handles=h, loc='lower center',
                          fancybox=True, shadow=True, ncol=4)
        else:
            plt.grid()
            means = [mean[:args.episode_length] for mean in means]
            stds = [std[:args.episode_length] for std in stds]
            # means = [means[2], means[3], means[0], means[1]]
            # stds = [stds[2], stds[3], stds[0], stds[1]]
            h = plot_performance_over_time_with_stds(ax,
                                                     range(args.episode_length),
                                                     means,
                                                     stds,
                                                     std_scale=0.25)
            plt.figlegend(labels=args.labels,
                          handles=h, loc='lower center',
                          fancybox=True, shadow=True, ncol=4)
        ax.title.set_text(function_name.capitalize())
        ax.set_ylim(0, 1)
        if i != 0 and i != 4:
            ax.set_yticklabels([])

        if i < 4:
            ax.set_xticklabels([])
        ax.grid(axis='y')
    # plt.legend(df[df['run_on'] == function_name]['trained_on'], loc=4)


def plot_performance_by_labels(ax, labels, performance, width=0.5):
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=25)
    return ax.bar(labels, performance, width, color=mcolors.TABLEAU_COLORS)


def plot_performance_over_time_with_stds(ax, x, means, stds, std_scale=0.1):
    h = []
    for i, (mean, std) in enumerate(zip(means, stds)):
        h += ax.plot(x, mean, color=COLORS[i])
        ax.fill_between(x, mean, mean - std_scale * std, color=COLORS[i], alpha=0.2)
        ax.fill_between(x, mean, mean + std_scale * std, color=COLORS[i], alpha=0.2)
    return h


def convert_series_to_array(series):
    return [np.array(element.strip("{").strip("}").split(","), np.float64) for element in series]


def get_data(args):
    df = read_from_sql(args.non_learned, args.algorithms, args.trained_ons, args.run_ids, args.operations, args.run_ons)
    if arguments.same_as_trained:
        df = df[df['trained_on'] == df['run_on']]
    base_group_by = ['algorithm', 'trained_on', 'run_id', 'step_counter']
    group_by = base_group_by + [arguments.plotby, ]
    group_by = df.groupby(group_by)
    group_names = list(group_by.groups)
    data = [group_name for group_name in group_names]
    means = []
    stds = []
    for name in group_names:
        group = group_by.get_group(name)
        means.append(np.mean(convert_series_to_array(group['mean_perf_over_time']), axis=0))
        stds.append(np.mean(convert_series_to_array(group['std_perf_over_time']), axis=0))
    means_and_stds = [[mean] + [std] for mean, std in zip(means, stds)]
    data = [list(d) + mean for d, mean in zip(data, means_and_stds)]
    columns = df.columns
    columns = columns.drop('index')
    if arguments.plotby == "run_on":
        columns = columns.drop('operation')

    return pd.DataFrame(data=data, columns=columns)


if __name__ == "__main__":
    if not os.path.isdir(PLOT_DIR):
        os.makedirs(PLOT_DIR)

    arg_parser = argparse.ArgumentParser(description="flag to include non-learned algorithms")
    arg_parser.add_argument("-name",
                            "--name",
                            dest="name",
                            help="provide plot title",
                            type=str,
                            default="1_obs-out-of-distribution-by-operator")
    arg_parser.add_argument("-n",
                            "--non-learned",
                            dest="non_learned",
                            help="provide number of parallel calls",
                            type=bool,
                            action=argparse.BooleanOptionalAction,
                            default=False)
    arg_parser.add_argument("-a",
                            "--algorithms",
                            dest="algorithms",
                            help="provide algorithm names",
                            nargs='+',
                            type=list,
                            default=["reinforce"])
    arg_parser.add_argument("-t",
                            "--trained_on",
                            dest="trained_ons",
                            help="provide trained on function names",
                            nargs='+',
                            type=list,
                            default=[name for name in FUNCTIONS.keys()])
    arg_parser.add_argument("-i",
                            "--run_id",
                            dest="run_ids",
                            help="provide run ids",
                            nargs='+',
                            type=list,
                            default=['1_obs', ])
    arg_parser.add_argument("-r",
                            "--run-ons",
                            dest="run_ons",
                            help="provide run on function names",
                            nargs='+',
                            type=list,
                            default=["all", ])
    arg_parser.add_argument("-o",
                            "--operations",
                            dest="operations",
                            help="provide operation names",
                            nargs='+',
                            type=list,
                            default=['all', ])
    arg_parser.add_argument("-l",
                            "--episode-length",
                            dest="episode_length",
                            help="provide plot episode length",
                            type=int,
                            default=50)
    arg_parser.add_argument("-sub",
                            "--subplots",
                            dest="subplots",
                            help="print every run on function in own subplot",
                            type=bool,
                            action=argparse.BooleanOptionalAction,
                            default=True)
    arg_parser.add_argument("-bar",
                            "--barplots",
                            dest="barplots",
                            help="print every run on function as a barplot",
                            type=bool,
                            action=argparse.BooleanOptionalAction,
                            default=True)
    arg_parser.add_argument("-by",
                            "--plotby",
                            dest="plotby",
                            help="if parameter subplots is set, define which attribute to split the plot by",
                            type=str,
                            default="operation")
    arg_parser.add_argument("-same",
                            "--same_as_trained",
                            dest="same_as_trained",
                            help="if parameter same_as_trained is set,"
                                 " only the runs, that are run on the same function as the agent has trained on,"
                                 " are being represented",
                            type=bool,
                            action=argparse.BooleanOptionalAction,
                            default=True)
    # arguments = arg_parser.parse_args()
    gin.parse_config_file("plot_performance_from_runs_table.gin")
    arguments = Args()
    main(arguments)
