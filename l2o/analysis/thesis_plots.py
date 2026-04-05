import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from objective_functions.tf_objective_functions import FUNCTIONS
from definitions import RUNS_DIR


def plot_free_parameter_comparison(
        one_free,
        half_free,
        pre_last_free,
        std1,
        std2,
        std3,
        plot_dir,
        input_dimension,
        function_name,
        std_scale=0.1):
    """Plots a comparison between different numbers of free parameters."""
    if function_name == "Styblinski_tang":
        function_name = "Styblinski-Tang"

    xx = np.arange(0, len(one_free))
    colors = [color for color in mcolors.TABLEAU_COLORS.values()]

    plt.figure(figsize=(10, 6))
    
    # 1 Free Parameter
    plt.plot(xx, one_free, color=colors[0], label="1 Free Parameter")
    plt.fill_between(xx, one_free - std_scale * std1, one_free + std_scale * std1, color=colors[0], alpha=0.2)

    # Half Free Parameters
    plt.plot(xx, half_free, color=colors[1], label=f"{int(input_dimension/2)} Free Parameters")
    plt.fill_between(xx, half_free - std_scale * std2, half_free + std_scale * std2, color=colors[1], alpha=0.2)

    # Input Dimension - 1 Free Parameters
    plt.plot(xx, pre_last_free, color=colors[2], label=f"{input_dimension-1} Free Parameters")
    plt.fill_between(xx, pre_last_free - std_scale * std3, pre_last_free + std_scale * std3, color=colors[2], alpha=0.2)

    ylabel_prefix = ""
    if function_name in ["Rosenbrock", "Zakharov"]:
        plt.yscale('log')
        ylabel_prefix = "Logarithmic "

    plt.legend(loc='upper right')
    plt.xlim(0, len(one_free))
    plt.ylim(bottom=0)
    plt.ylabel(f"{ylabel_prefix}Squared Error")
    plt.xlabel("Training Iteration Step")
    plt.title(f"{input_dimension}D {function_name} MSE Performance Comparison")
    plt.grid(True)
    
    filename = f"{ylabel_prefix}MSE_{function_name}_free_comparison.png"
    plt.savefig(os.path.join(plot_dir, filename), transparent=True)
    plt.close()


def plot_function_comparison_2d(performances_dict, stds_dict, plot_dir, std_scale=0.1):
    """Plots a comparison of different benchmark functions on a single plot."""
    plt.figure(figsize=(12, 8))
    xx = None
    colors = list(mcolors.TABLEAU_COLORS.values())

    for i, (function_name, perf) in enumerate(performances_dict.items()):
        if xx is None:
            xx = np.arange(0, len(perf))
        
        std = stds_dict.get(function_name, np.zeros_like(perf))
        plt.plot(xx, perf, color=colors[i % len(colors)], label=function_name.capitalize())
        plt.fill_between(xx, perf - std_scale * std, perf + std_scale * std, color=colors[i % len(colors)], alpha=0.2)

    plt.legend(loc='upper right')
    plt.xlim(0, len(xx))
    plt.ylim(0, 1.1)
    plt.ylabel("Performance")
    plt.xlabel("Training Iteration Step")
    plt.title("Benchmark Function Performance Comparison (2D)")
    plt.grid(True)
    plt.savefig(os.path.join(plot_dir, "function_comparison_2d.png"), transparent=True)
    plt.close()


if __name__ == "__main__":
    # This script can be expanded to load data and generate thesis-style plots
    print("Thesis plots script initialized.")
