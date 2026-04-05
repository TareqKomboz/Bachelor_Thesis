import os
import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.colors as mcolors

from objective_functions.tf_objective_functions import FUNCTIONS
from evaluation.evaluation_utils import build_evaluation_parameters
from common.utils import load_array_from_csv, save_data
from definitions import RUNS_DIR


def calc_mse(function_values, optimal_values):
    """Calculates Mean Squared Error and Standard Deviation."""
    squared_error = (function_values - optimal_values) ** 2
    return np.mean(squared_error), np.std(squared_error)


def plot_mse(
        mse,
        std,
        plot_dir,
        input_dimension,
        number_free_parameters,
        function_name,
        std_scale=0.1):
    """Plots the Mean Squared Error over training steps."""
    if function_name == "Styblinski_tang":
        function_name = "Styblinski-Tang"

    xx = np.arange(0, len(mse))
    colors = [color for color in mcolors.TABLEAU_COLORS.values()]

    plt.plot(xx, mse, color=colors[0])
    plt.fill_between(xx, mse, mse - std_scale * std, color=colors[0], alpha=0.2)
    plt.fill_between(xx, mse, mse + std_scale * std, color=colors[0], alpha=0.2)

    plt.xlim(0, len(mse))
    plt.ylim(0)
    plt.ylabel("Squared Error")
    plt.xlabel("Training Iteration Step")
    plt.title("{}D {} with {} Free Parameter - MSE Performance".format(
        input_dimension, function_name, number_free_parameters))
    plt.grid()
    plt.savefig(os.path.join(plot_dir, "mse.png"), transparent=True)
    plt.clf()


def run_mse_calculation(input_dimension, n_start_pos=4):
    """Computes MSE for a given dimension and saves the results."""
    batch_size = n_start_pos ** input_dimension
    initial_state = build_evaluation_parameters(n_start_pos=n_start_pos, input_dimension=input_dimension)
    
    # Example logic for a single dimension and parameter configuration
    number_free_parameters = 1
    free_values = initial_state[:, :number_free_parameters]
    transposed_free_values = tf.transpose(free_values)
    rest_zeros = tf.zeros(shape=(input_dimension - number_free_parameters, batch_size))
    rest_ones = tf.ones(shape=(input_dimension - number_free_parameters, batch_size))
    
    # This is a generic setup for Ackley, could be expanded for other functions
    for function_name in ["Ackley"]:
        base_directory = os.path.join(
            RUNS_DIR, 
            "agent_name_ppo", 
            f"input_dimension_{input_dimension}", 
            f"number_free_parameters_{number_free_parameters}", 
            function_name,
            "abs_env_50_epsLen_1_numObs" # Example run_id
        )
        
        if not os.path.isdir(base_directory):
            print(f"Directory not found: {base_directory}")
            continue

        objective_function = FUNCTIONS[function_name]
        
        # Max input at bounds
        max_input_x = tf.concat(
            values=[tf.convert_to_tensor(transposed_free_values), tf.convert_to_tensor(-1 * rest_ones)],
            axis=0
        )
        my_max = objective_function(x=max_input_x).numpy()
        
        # Known optimal for these functions is usually at 0 or 1
        min_input_opt = rest_zeros
        min_input_x = tf.concat(
            values=[tf.convert_to_tensor(transposed_free_values), tf.convert_to_tensor(min_input_opt)],
            axis=0
        )
        my_min = objective_function(x=min_input_x).numpy()

        csv_path = os.path.join(base_directory, "CSVs", "train_rewards.csv")
        if not os.path.isfile(csv_path):
            print(f"CSV not found: {csv_path}")
            continue

        train_rewards = load_array_from_csv(csv_path, ',')

        my_mse_list = []
        my_std_list = []
        for reward in train_rewards:
            # Denormalize reward to get function values
            function_values = my_max - (reward * (my_max - my_min))
            mse_val, std_val = calc_mse(function_values, my_min)
            my_mse_list.append(mse_val)
            my_std_list.append(std_val)

        mse_array = np.asarray(a=my_mse_list, dtype=np.float32)
        std_array = np.asarray(a=my_std_list, dtype=np.float32)

        save_data(
            data={
                os.path.join(base_directory, "CSVs", "mse_over_batches.csv"): mse_array,
                os.path.join(base_directory, "CSVs", "std_mse_over_batches.csv"): std_array,
            }
        )

        plot_mse(
            mse=mse_array,
            std=std_array,
            plot_dir=base_directory,
            input_dimension=input_dimension,
            number_free_parameters=number_free_parameters,
            function_name=function_name
        )


if __name__ == "__main__":
    # Example usage
    run_mse_calculation(input_dimension=10)
