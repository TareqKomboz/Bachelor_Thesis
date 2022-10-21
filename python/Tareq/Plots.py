from python.objective_functions.tf_objective_functions import FUNCTIONS
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.colors as mcolors


def load_array_from_csv(csv_filename, delimiter=','):
    return np.loadtxt(csv_filename, delimiter=delimiter)


def my_plot_rest(
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
    if function_name == "Styblinski_tang":
        function_name = "Styblinski-Tang"

    xx = np.arange(0, len(one_free))

    colors = [color for color in mcolors.TABLEAU_COLORS.values()]

    plt.plot(xx, one_free, color=colors[0], label="1 Free Parameter")
    plt.fill_between(xx, one_free, one_free - std_scale * std1, color=colors[0], alpha=0.2)
    plt.fill_between(xx, one_free, one_free + std_scale * std1, color=colors[0], alpha=0.2)

    plt.plot(xx, half_free, color=colors[1], label="{} Free Parameters".format(int(input_dimension/2)))
    plt.fill_between(xx, half_free, half_free - std_scale * std2, color=colors[1], alpha=0.2)
    plt.fill_between(xx, half_free, half_free + std_scale * std2, color=colors[1], alpha=0.2)

    plt.plot(xx, pre_last_free, color=colors[2], label="{} Free Parameters".format(input_dimension-1))
    plt.fill_between(xx, pre_last_free, pre_last_free - std_scale * std3, color=colors[2], alpha=0.2)
    plt.fill_between(xx, pre_last_free, pre_last_free + std_scale * std3, color=colors[2], alpha=0.2)

    add_string = ""
    if function_name == "Rosenbrock" or function_name == "Zakharov":
        plt.yscale('log')
        add_string = "Logarithmic "

    plt.legend(loc='upper right')

    plt.xlim(0, 10000)
    plt.ylim(0)
    plt.ylabel(add_string+"Squared Error")
    plt.xlabel("Training Iteration Step")
    plt.title("{}D {} MSE Performance".format(input_dimension, function_name))
    plt.grid()
    plt.savefig((plot_dir + "\\"+add_string+"MSE_{}_free_comparison.png".format(function_name)), transparent=True)
    plt.clf()


def function_comparison_2d(
        ackley,
        griewank,
        levy,
        rastrigin,
        rosenbrock,
        sphere,
        styblinski_tang,
        zakharov,
        std1,
        std2,
        std3,
        std4,
        std5,
        std6,
        std7,
        std8,
        plot_dir,
        std_scale=0.1):
    xx = np.arange(0, len(ackley))

    colors = [color for color in mcolors.TABLEAU_COLORS.values()]

    plt.plot(xx, ackley, color=colors[0], label="Ackley")
    plt.fill_between(xx, ackley, ackley - std_scale * std1, color=colors[0], alpha=0.2)
    plt.fill_between(xx, ackley, ackley + std_scale * std1, color=colors[0], alpha=0.2)

    plt.plot(xx, griewank, color=colors[1], label="Griewank")
    plt.fill_between(xx, griewank, griewank - std_scale * std2, color=colors[1], alpha=0.2)
    plt.fill_between(xx, griewank, griewank + std_scale * std2, color=colors[1], alpha=0.2)

    plt.plot(xx, levy, color=colors[2], label="Levy")
    plt.fill_between(xx, levy, levy - std_scale * std3, color=colors[2], alpha=0.2)
    plt.fill_between(xx, levy, levy + std_scale * std3, color=colors[2], alpha=0.2)

    plt.plot(xx, rastrigin, color=colors[3], label="Rastrigin")
    plt.fill_between(xx, rastrigin, rastrigin - std_scale * std4, color=colors[3], alpha=0.2)
    plt.fill_between(xx, rastrigin, rastrigin + std_scale * std4, color=colors[3], alpha=0.2)

    plt.plot(xx, rosenbrock, color=colors[4], label="Rosenbrock")
    plt.fill_between(xx, rosenbrock, rosenbrock - std_scale * std5, color=colors[4], alpha=0.2)
    plt.fill_between(xx, rosenbrock, rosenbrock + std_scale * std5, color=colors[4], alpha=0.2)

    plt.plot(xx, sphere, color=colors[5], label="Sphere")
    plt.fill_between(xx, sphere, sphere - std_scale * std6, color=colors[5], alpha=0.2)
    plt.fill_between(xx, sphere, sphere + std_scale * std6, color=colors[5], alpha=0.2)

    plt.plot(xx, styblinski_tang, color=colors[6], label="Styblinski-Tang")
    plt.fill_between(xx, styblinski_tang, styblinski_tang - std_scale * std7, color=colors[6], alpha=0.2)
    plt.fill_between(xx, styblinski_tang, styblinski_tang + std_scale * std7, color=colors[6], alpha=0.2)

    plt.plot(xx, zakharov, color=colors[7], label="Zakharov")
    plt.fill_between(xx, zakharov, zakharov - std_scale * std8, color=colors[7], alpha=0.2)
    plt.fill_between(xx, zakharov, zakharov + std_scale * std8, color=colors[7], alpha=0.2)

    plt.yscale('log')

    plt.legend(loc='upper right')

    plt.xlim(0, 10000)
    plt.ylim(0)
    plt.ylabel("Logarithmic Squared Error")
    plt.xlabel("Training Iteration Step")
    plt.title("2D MSE Performance with 1 Free Parameter")
    plt.grid()
    plt.savefig((plot_dir + "\\Logarithmic MSE_function_comparison.png"), transparent=True)
    plt.clf()


def free_fixed(
        dim2,
        dim4,
        dim6,
        dim8,
        dim10,
        std1,
        std2,
        std3,
        std4,
        std5,
        plot_dir,
        function_name,
        std_scale=0.1):
    if function_name == "Styblinski_tang":
        function_name = "Styblinski-Tang"

    xx = np.arange(0, len(dim2))

    colors = [color for color in mcolors.TABLEAU_COLORS.values()]

    plt.plot(xx, dim2, color=colors[0], label="Input Dimension 2 - 1 Free")
    plt.fill_between(xx, dim2, dim2 - std_scale * std1, color=colors[0], alpha=0.2)
    plt.fill_between(xx, dim2, dim2 + std_scale * std1, color=colors[0], alpha=0.2)

    plt.plot(xx, dim4, color=colors[1], label="Input Dimension 4 - 3 Free")
    plt.fill_between(xx, dim4, dim4 - std_scale * std2, color=colors[1], alpha=0.2)
    plt.fill_between(xx, dim4, dim4 + std_scale * std2, color=colors[1], alpha=0.2)

    plt.plot(xx, dim6, color=colors[2], label="Input Dimension 6 - 5 Free")
    plt.fill_between(xx, dim6, dim6 - std_scale * std3, color=colors[2], alpha=0.2)
    plt.fill_between(xx, dim6, dim6 + std_scale * std3, color=colors[2], alpha=0.2)

    plt.plot(xx, dim8, color=colors[3], label="Input Dimension 8 - 7 Free")
    plt.fill_between(xx, dim8, dim8 - std_scale * std4, color=colors[3], alpha=0.2)
    plt.fill_between(xx, dim8, dim8 + std_scale * std4, color=colors[3], alpha=0.2)

    plt.plot(xx, dim10, color=colors[4], label="Input Dimension 10 - 9 Free")
    plt.fill_between(xx, dim10, dim10 - std_scale * std5, color=colors[4], alpha=0.2)
    plt.fill_between(xx, dim10, dim10 + std_scale * std5, color=colors[4], alpha=0.2)

    add_string = ""
    if function_name == "Rosenbrock" or function_name == "Zakharov":
        plt.yscale('log')
        add_string = "Logarithmic "

    plt.legend(loc='upper right')

    plt.xlim(0, 10000)
    plt.ylim(0)
    plt.ylabel(add_string+"Squared Error")
    plt.xlabel("Training Iteration Step")
    plt.title("{} MSE Performance with D-1 Free Parameters".format(function_name))
    plt.grid()
    plt.savefig((plot_dir + "\\"+add_string+"MSE_{}_fixed_d-1_free.png".format(function_name)), transparent=True)
    plt.clf()


def compare_rnn(
        nn,
        rnn,
        std1,
        std2,
        plot_dir,
        input_dimension,
        number_free_parameters,
        function_name,
        std_scale=0.1):
    if function_name == "Styblinski_tang":
        function_name = "Styblinski-Tang"

    xx = np.arange(0, len(nn))

    colors = [color for color in mcolors.TABLEAU_COLORS.values()]

    plt.plot(xx, nn, color=colors[0], label="Neural Network")
    plt.fill_between(xx, nn, nn - std_scale * std1, color=colors[0], alpha=0.2)
    plt.fill_between(xx, nn, nn + std_scale * std1, color=colors[0], alpha=0.2)

    plt.plot(xx, rnn, color=colors[1], label="Recurrent Neural Network")
    plt.fill_between(xx, rnn, rnn - std_scale * std2, color=colors[1], alpha=0.2)
    plt.fill_between(xx, rnn, rnn + std_scale * std2, color=colors[1], alpha=0.2)

    plt.legend(loc='upper right')

    plt.xlim(0, 10000)
    plt.ylim(0)
    plt.ylabel("Squared Error")
    plt.xlabel("Training Iteration Step")
    plt.title("{}D {} MSE Performance with {} Free Parameters".format(input_dimension, function_name, number_free_parameters))
    plt.grid()
    plt.savefig((plot_dir + "\\MSE_{}D_{}_{}_free_rnn_comparison.png".format(input_dimension, function_name, number_free_parameters)), transparent=True)
    plt.clf()


if __name__ == "__main__":
    # i = 0
    # for function_name in FUNCTIONS.keys():
    #     for input_dimension in [4, 6, 8, 10]:
    #         base_directory = "C:\\Users\\Zeyad\\PycharmProjects\\Kilian_modified\\linux\\runs\\input_dimension_{0}".format(
    #             input_dimension
    #         )
    #
    #         one_free = load_array_from_csv(csv_filename=base_directory+"\\number_free_parameters_{0}\\{1}\\abs_reinforce_env_50_epsLen_1_numObs\\CSVs\\mse_over_batches.csv".format(1, function_name))
    #         half_free = load_array_from_csv(csv_filename=base_directory+"\\number_free_parameters_{0}\\{1}\\abs_reinforce_env_50_epsLen_1_numObs\\CSVs\\mse_over_batches.csv".format(int(input_dimension/2), function_name))
    #         pre_last_free = load_array_from_csv(csv_filename=base_directory+"\\number_free_parameters_{0}\\{1}\\abs_reinforce_env_50_epsLen_1_numObs\\CSVs\\mse_over_batches.csv".format(input_dimension-1, function_name))
    #
    #         std1 = load_array_from_csv(
    #             csv_filename=base_directory + "\\number_free_parameters_{0}\\{1}\\abs_reinforce_env_50_epsLen_1_numObs\\CSVs\\std_mse_over_batches.csv".format(
    #                 1, function_name))
    #         std2 = load_array_from_csv(
    #             csv_filename=base_directory + "\\number_free_parameters_{0}\\{1}\\abs_reinforce_env_50_epsLen_1_numObs\\CSVs\\std_mse_over_batches.csv".format(
    #                 int(input_dimension / 2), function_name))
    #         std3 = load_array_from_csv(
    #             csv_filename=base_directory + "\\number_free_parameters_{0}\\{1}\\abs_reinforce_env_50_epsLen_1_numObs\\CSVs\\std_mse_over_batches.csv".format(
    #                 input_dimension - 1, function_name))
    #
    #         my_plot_rest(
    #             one_free=one_free,
    #             half_free=half_free,
    #             pre_last_free=pre_last_free,
    #             std1=std1,
    #             std2=std2,
    #             std3=std3,
    #             plot_dir=base_directory,
    #             input_dimension=input_dimension,
    #             function_name=function_name
    #         )
    #         i += 1
    #         print("wow {}".format(i))
    #
    i = 0
    base_directory = "C:\\Users\\Zeyad\\PycharmProjects\\Kilian_modified\\linux\\runs\\input_dimension_2"
    function_comparison_2d(
        ackley=load_array_from_csv(
            csv_filename=base_directory + "\\number_free_parameters_{0}\\{1}\\abs_reinforce_env_50_epsLen_1_numObs\\CSVs\\mse_over_batches.csv".format(
                1, "Ackley")),
        griewank=load_array_from_csv(
            csv_filename=base_directory + "\\number_free_parameters_{0}\\{1}\\abs_reinforce_env_50_epsLen_1_numObs\\CSVs\\mse_over_batches.csv".format(
                1, "Griewank")),
        levy=load_array_from_csv(
            csv_filename=base_directory + "\\number_free_parameters_{0}\\{1}\\abs_reinforce_env_50_epsLen_1_numObs\\CSVs\\mse_over_batches.csv".format(
                1, "Levy")),
        rastrigin=load_array_from_csv(
            csv_filename=base_directory + "\\number_free_parameters_{0}\\{1}\\abs_reinforce_env_50_epsLen_1_numObs\\CSVs\\mse_over_batches.csv".format(
                1, "Rastrigin")),
        rosenbrock=load_array_from_csv(
            csv_filename=base_directory + "\\number_free_parameters_{0}\\{1}\\abs_reinforce_env_50_epsLen_1_numObs\\CSVs\\mse_over_batches.csv".format(
                1, "Rosenbrock")),
        sphere=load_array_from_csv(
            csv_filename=base_directory + "\\number_free_parameters_{0}\\{1}\\abs_reinforce_env_50_epsLen_1_numObs\\CSVs\\mse_over_batches.csv".format(
                1, "Sphere")),
        styblinski_tang=load_array_from_csv(
            csv_filename=base_directory + "\\number_free_parameters_{0}\\{1}\\abs_reinforce_env_50_epsLen_1_numObs\\CSVs\\mse_over_batches.csv".format(
                1, "Styblinski_tang")),
        zakharov=load_array_from_csv(
            csv_filename=base_directory + "\\number_free_parameters_{0}\\{1}\\abs_reinforce_env_50_epsLen_1_numObs\\CSVs\\mse_over_batches.csv".format(
                1, "Zakharov")),
        std1=load_array_from_csv(
            csv_filename=base_directory + "\\number_free_parameters_{0}\\{1}\\abs_reinforce_env_50_epsLen_1_numObs\\CSVs\\std_mse_over_batches.csv".format(
                1, "Ackley")),
        std2=load_array_from_csv(
            csv_filename=base_directory + "\\number_free_parameters_{0}\\{1}\\abs_reinforce_env_50_epsLen_1_numObs\\CSVs\\std_mse_over_batches.csv".format(
                1, "Griewank")),
        std3=load_array_from_csv(
            csv_filename=base_directory + "\\number_free_parameters_{0}\\{1}\\abs_reinforce_env_50_epsLen_1_numObs\\CSVs\\std_mse_over_batches.csv".format(
                1, "Levy")),
        std4=load_array_from_csv(
            csv_filename=base_directory + "\\number_free_parameters_{0}\\{1}\\abs_reinforce_env_50_epsLen_1_numObs\\CSVs\\std_mse_over_batches.csv".format(
                1, "Rastrigin")),
        std5=load_array_from_csv(
            csv_filename=base_directory + "\\number_free_parameters_{0}\\{1}\\abs_reinforce_env_50_epsLen_1_numObs\\CSVs\\std_mse_over_batches.csv".format(
                1, "Rosenbrock")),
        std6=load_array_from_csv(
            csv_filename=base_directory + "\\number_free_parameters_{0}\\{1}\\abs_reinforce_env_50_epsLen_1_numObs\\CSVs\\std_mse_over_batches.csv".format(
                1, "Sphere")),
        std7=load_array_from_csv(
            csv_filename=base_directory + "\\number_free_parameters_{0}\\{1}\\abs_reinforce_env_50_epsLen_1_numObs\\CSVs\\std_mse_over_batches.csv".format(
                1, "Styblinski_tang")),
        std8=load_array_from_csv(
            csv_filename=base_directory + "\\number_free_parameters_{0}\\{1}\\abs_reinforce_env_50_epsLen_1_numObs\\CSVs\\std_mse_over_batches.csv".format(
                1, "Zakharov")),
        plot_dir=base_directory
    )
    i += 1
    print("wow {}".format(i))

    # i = 0
    # for function_name in FUNCTIONS.keys():
    #     base_directory = "C:\\Users\\Zeyad\\PycharmProjects\\Kilian_modified\\linux\\runs"
    #
    #     dim2 = load_array_from_csv(
    #         csv_filename=base_directory+"\\input_dimension_2\\number_free_parameters_{0}\\{1}\\abs_reinforce_env_50_epsLen_1_numObs\\CSVs\\mse_over_batches.csv".format(
    #             1,
    #             function_name
    #         ))
    #     dim4 = load_array_from_csv(
    #         csv_filename=base_directory+"\\input_dimension_4\\number_free_parameters_{0}\\{1}\\abs_reinforce_env_50_epsLen_1_numObs\\CSVs\\mse_over_batches.csv".format(
    #             3,
    #             function_name
    #     ))
    #     dim6 = load_array_from_csv(
    #         csv_filename=base_directory+"\\input_dimension_6\\number_free_parameters_{0}\\{1}\\abs_reinforce_env_50_epsLen_1_numObs\\CSVs\\mse_over_batches.csv".format(
    #             5,
    #             function_name
    #         ))
    #     dim8 = load_array_from_csv(
    #         csv_filename=base_directory+"\\input_dimension_8\\number_free_parameters_{0}\\{1}\\abs_reinforce_env_50_epsLen_1_numObs\\CSVs\\mse_over_batches.csv".format(
    #             7,
    #             function_name
    #         ))
    #     dim10 = load_array_from_csv(
    #         csv_filename=base_directory+"\\input_dimension_10\\number_free_parameters_{0}\\{1}\\abs_reinforce_env_50_epsLen_1_numObs\\CSVs\\std_mse_over_batches.csv".format(
    #             9,
    #             function_name
    #         ))
    #
    #     std1 = load_array_from_csv(
    #         csv_filename=base_directory + "\\input_dimension_2\\number_free_parameters_{0}\\{1}\\abs_reinforce_env_50_epsLen_1_numObs\\CSVs\\std_mse_over_batches.csv".format(
    #             1,
    #             function_name
    #         ))
    #     std2 = load_array_from_csv(
    #         csv_filename=base_directory + "\\input_dimension_4\\number_free_parameters_{0}\\{1}\\abs_reinforce_env_50_epsLen_1_numObs\\CSVs\\std_mse_over_batches.csv".format(
    #             3,
    #             function_name
    #         ))
    #     std3 = load_array_from_csv(
    #         csv_filename=base_directory + "\\input_dimension_6\\number_free_parameters_{0}\\{1}\\abs_reinforce_env_50_epsLen_1_numObs\\CSVs\\std_mse_over_batches.csv".format(
    #             5,
    #             function_name
    #         ))
    #     std4 = load_array_from_csv(
    #         csv_filename=base_directory + "\\input_dimension_8\\number_free_parameters_{0}\\{1}\\abs_reinforce_env_50_epsLen_1_numObs\\CSVs\\std_mse_over_batches.csv".format(
    #             7,
    #             function_name
    #         ))
    #     std5 = load_array_from_csv(
    #         csv_filename=base_directory + "\\input_dimension_10\\number_free_parameters_{0}\\{1}\\abs_reinforce_env_50_epsLen_1_numObs\\CSVs\\std_mse_over_batches.csv".format(
    #             9,
    #             function_name
    #         ))
    #
    #     free_fixed(
    #         dim2=dim2,
    #         dim4=dim4,
    #         dim6=dim6,
    #         dim8=dim8,
    #         dim10=dim10,
    #         std1=std1,
    #         std2=std2,
    #         std3=std3,
    #         std4=std4,
    #         std5=std5,
    #         plot_dir=base_directory,
    #         function_name=function_name
    #     )
    #     i += 1
    #     print("wow {}".format(i))

    # i = 0
    # for input_dimension in [2, 10]:
    #     for number_free_parameters in [1, input_dimension-1]:
    #         for function_name in ["Ackley", "Griewank", "Rastrigin"]:
    #             base_directory = "C:\\Users\\Zeyad\\PycharmProjects\\Kilian_modified\\linux"
    #
    #             nn = load_array_from_csv(csv_filename=base_directory+"\\runs\\input_dimension_{0}\\number_free_parameters_{1}\\{2}\\abs_reinforce_env_50_epsLen_1_numObs\\CSVs\\mse_over_batches.csv".format(
    #                 input_dimension,
    #                 number_free_parameters,
    #                 function_name
    #             ))
    #             rnn = load_array_from_csv(csv_filename=base_directory+"\\runs_rnn\\agent_name_rnn_reinforce\\input_dimension_{0}\\number_free_parameters_{1}\\{2}\\abs_env_50_epsLen_1_numObs\\CSVs\\mse_over_batches.csv".format(
    #                 input_dimension,
    #                 number_free_parameters,
    #                 function_name
    #             ))
    #
    #             compare_rnn(
    #                 nn=nn,
    #                 rnn=rnn,
    #                 plot_dir=base_directory,
    #                 input_dimension=input_dimension,
    #                 number_free_parameters=number_free_parameters,
    #                 function_name=function_name
    #             )
    #             i += 1
    #             print("wow {}".format(i))
