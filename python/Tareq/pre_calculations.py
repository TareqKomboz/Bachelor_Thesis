from python.objective_functions.tf_objective_functions import FUNCTIONS
from python.evaluation.evaluation_utils import build_evaluation_parameters
import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.colors as mcolors


def save_array_as_csv(csv_filename, data_array, delimiter=','):
    np.savetxt(csv_filename, data_array, delimiter=delimiter)


def load_array_from_csv(csv_filename, delimiter=','):
    return np.loadtxt(csv_filename, delimiter=delimiter)


def save_data(data):
    for key in data:
        save_array_as_csv(
            csv_filename=key,
            data_array=data[key]
        )


def calc_mse(function_values, optimal_values):
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
    plt.title("{}D {} - {} Free - Mean Squared Error".format(input_dimension, function_name, number_free_parameters))
    plt.grid()
    plt.savefig((plot_dir + "\\mse.png"), transparent=True)
    plt.clf()


if __name__ == "__main__":
    n_start_pos = 4
    lmnop = 1
    for input_dimension in [10]:  # , 4, 6, 8, 10]:
        batch_size = n_start_pos ** input_dimension
        initial_state = build_evaluation_parameters(n_start_pos=n_start_pos, input_dimension=input_dimension)
        if input_dimension == 2:
            number_free_parameters = 1
            free_values = initial_state[:, :number_free_parameters]
            transposed_free_values = tf.transpose(free_values)
            rest_zeros = tf.zeros(shape=(input_dimension - number_free_parameters, batch_size))
            rest_ones = tf.ones(shape=(input_dimension - number_free_parameters, batch_size))
            max_input_x = tf.concat(
                values=[tf.convert_to_tensor(transposed_free_values), tf.convert_to_tensor(-1 * rest_ones)],
                axis=0
            )
            for function_name in FUNCTIONS.keys():
                base_directory = "C:\\Users\\Zeyad\\PycharmProjects\\Kilian_modified\\linux\\runs_long\\agent_name_reinforce\\input_dimension_{0}\\number_free_parameters_{1}\\{2}\\abs_env_50_epsLen_1_numObs".format(
                    input_dimension,
                    number_free_parameters,
                    function_name
                )
                objective_function = FUNCTIONS[function_name]
                my_max = objective_function(x=max_input_x).numpy()  # calc max
                if function_name == "Ackley" \
                        or function_name == "Griewank" \
                        or function_name == "Rastrigin" \
                        or function_name == "Sphere" \
                        or function_name == "Zakharov":
                    min_input_opt = rest_zeros
                elif function_name == "Levy":
                    min_input_opt = rest_ones / 10.0
                elif function_name == "Rosenbrock":
                    min_input_opt = rest_ones / 2.0
                elif function_name == "Styblinski_tang":
                    my_min = -39.16599 * input_dimension
                    min_input_opt = (rest_ones * -2.903534) / 5.0

                min_input_x = tf.concat(
                    values=[tf.convert_to_tensor(transposed_free_values), tf.convert_to_tensor(min_input_opt)],
                    axis=0
                )
                my_min = objective_function(x=min_input_x).numpy()

                train_rewards = load_array_from_csv(
                    base_directory + "\\CSVs\\train_rewards.csv",
                    ','
                )

                my_mse_list = []
                my_std_list = []
                for reward in train_rewards:
                    function_values = my_max - (reward * (my_max - my_min))
                    mse_val, std_val = calc_mse(function_values, my_min)
                    my_mse_list.append(mse_val)
                    my_std_list.append(std_val)

                mse_array = np.asarray(a=my_mse_list, dtype=np.float32)
                std_array = np.asarray(a=my_std_list, dtype=np.float32)

                save_data(
                    data={
                        base_directory + "\\CSVs\\mse_over_batches.csv": mse_array,
                        base_directory + "\\CSVs\\std_mse_over_batches.csv": std_array,
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
                print("wow {}".format(lmnop))
                lmnop += 1
        else:
            for number_free_parameters in [1]:  # , int((input_dimension / 2)), (input_dimension - 1)]:
                free_values = initial_state[:, :number_free_parameters]
                transposed_free_values = tf.transpose(free_values)
                rest_zeros = tf.zeros(shape=(input_dimension - number_free_parameters, batch_size))
                rest_ones = tf.ones(shape=(input_dimension - number_free_parameters, batch_size))
                max_input_x = tf.concat(
                    values=[tf.convert_to_tensor(transposed_free_values), tf.convert_to_tensor(-1 * rest_ones)],
                    axis=0
                )
                for function_name in ["Styblinski_tang"]:  # FUNCTIONS.keys():
                    base_directory = "C:\\Users\\Zeyad\\PycharmProjects\\Kilian_modified\\linux\\runs_long\\agent_name_reinforce\\input_dimension_{0}\\number_free_parameters_{1}\\{2}\\abs_env_50_epsLen_1_numObs".format(
                        input_dimension,
                        number_free_parameters,
                        function_name
                    )
                    objective_function = FUNCTIONS[function_name]
                    my_max = objective_function(x=max_input_x).numpy()  # calc max
                    if function_name == "Ackley" \
                            or function_name == "Griewank" \
                            or function_name == "Rastrigin" \
                            or function_name == "Sphere" \
                            or function_name == "Zakharov":
                        min_input_opt = rest_zeros
                    elif function_name == "Levy":
                        min_input_opt = rest_ones / 10.0
                    elif function_name == "Rosenbrock":
                        min_input_opt = rest_ones / 2.0
                    elif function_name == "Styblinski_tang":
                        my_min = -39.16599 * input_dimension
                        min_input_opt = (rest_ones * -2.903534) / 5.0

                    min_input_x = tf.concat(
                        values=[tf.convert_to_tensor(transposed_free_values), tf.convert_to_tensor(min_input_opt)],
                        axis=0
                    )
                    my_min = objective_function(x=min_input_x).numpy()

                    train_rewards = load_array_from_csv(
                        base_directory + "\\CSVs\\train_rewards.csv",
                        ','
                    )

                    my_mse_list = []
                    my_std_list = []
                    for reward in train_rewards:
                        function_values = my_max - (reward * (my_max - my_min))
                        mse_val, std_val = calc_mse(function_values, my_min)
                        my_mse_list.append(mse_val)
                        my_std_list.append(std_val)

                    mse_array = np.asarray(a=my_mse_list, dtype=np.float32)
                    std_array = np.asarray(a=my_std_list, dtype=np.float32)

                    save_data(
                        data={
                            base_directory + "\\CSVs\\mse_over_batches.csv": mse_array,
                            base_directory + "\\CSVs\\std_mse_over_batches.csv": std_array,
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
                    print("wow {}".format(lmnop))
                    lmnop += 1
