import gin
import matplotlib.pyplot as plt
from matplotlib import cm
from numpy import arange, meshgrid, array
import tensorflow as tf

from python.objective_functions.tf_objective_functions import FUNCTIONS


@gin.configurable()
def visualize_environment(objective_function, objective_function_name, parameter_bounds, input_dimension, font):
    # define range for input
    parameter_bounds = parameter_bounds
    r_min, r_max = parameter_bounds[0], parameter_bounds[1]
    font_axis_labels = font["font_axis_labels"]
    if input_dimension == 1:
        # sample input range uniformly at 0.1 increments
        x = tf.experimental.numpy.arange(r_min, r_max, 0.000001, dtype=tf.float32)
        output = objective_function(x=array([x]))

        # setting the axes at the centre
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        ax.spines['left'].set_position('center')
        ax.spines['bottom'].set_position('zero')
        ax.spines['right'].set_color('none')
        ax.spines['top'].set_color('none')
        ax.xaxis.set_ticks_position('bottom')
        ax.yaxis.set_ticks_position('left')

        # plot the function
        plt.plot(x, output, 'r')
        plt.title("{0} Function".format(objective_function_name), fontdict=font["font_title"])
        plt.xlabel("$x$", fontdict=font_axis_labels)
        plt.ylabel("$f(x)$", fontdict=font_axis_labels)
        plt.savefig("PNGs/Graphs/{0} Function.png".format(objective_function_name))
        plt.show()
    elif input_dimension == 2:
        x_axis = tf.experimental.numpy.arange(r_min, r_max, 0.001, dtype=tf.float32)
        y_axis = tf.experimental.numpy.arange(r_min, r_max, 0.001, dtype=tf.float32)

        x, y = tf.meshgrid(x_axis, y_axis)

        results = objective_function(x=array([x, y]))

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_surface(X=x, Y=y, Z=results, cmap=cm.Blues)

        plt.title("{0} Function".format(objective_function_name), fontdict=font["font_title"])
        plt.xlabel("x", fontdict=font_axis_labels)
        plt.ylabel("y", fontdict=font_axis_labels)
        # plt.set_zlabel("$f(x_1,x_2)$", fontdict=font_axis_labels)

        # figure.colorbar(surf, shrink=0.5, aspect=5)
        plt.savefig("PNGs/Graphs/{0} Function.png".format(objective_function_name))
        plt.show()


@gin.configurable(denylist=['objective_function_name', 'metric', 'metric_as_string'])
def visualize_metric(objective_function_name,
                     metric,
                     metric_as_string,
                     number_training_iterations,
                     evaluation_interval,
                     input_dimension,
                     font):
    steps = range(0, number_training_iterations + 1, evaluation_interval)
    plt.plot(steps, metric)
    plt.title(
        "Experimental Results {0}d {1}".format(input_dimension + 1, objective_function_name),
        fontdict=font["font_title"]
    )
    font_axis_labels = font["font_axis_labels"]
    plt.xlabel("Number of Trainings Iterations", fontdict=font_axis_labels)
    plt.ylabel("{0}".format(metric_as_string), fontdict=font_axis_labels)
    plt.savefig("PNGs/Experimental Results/Experimental Results {0} {1}d {2}.png".format(
        metric_as_string,
        input_dimension + 1,
        objective_function_name
    ))
    plt.show()


def visualize(dataset, average_final_objective_function_values, average_episode_rewards):
    visualize_environment(dataset=dataset)
    visualize_metric(
        objective_function_name=dataset.objective_function_name,
        metric=average_final_objective_function_values,
        metric_as_string="Average Final Objective Function Value"
    )
    visualize_metric(
        objective_function_name=dataset.objective_function_name,
        metric=average_episode_rewards,
        metric_as_string="Average Episode Rewards"
    )


if __name__ == "__main__":
    # gin.parse_config_file('visualization_config.gin')
    # visualize(
    #     dataset=,
    #     average_final_objective_function_values=array([]),
    #     average_episode_rewards=array([])
    # )

    for objective_function_name, fct_tuple in FUNCTIONS.items():
        objective_function = fct_tuple[2]
        visualize_environment(
            objective_function=objective_function,
            objective_function_name=objective_function_name,
            parameter_bounds=(-1.0, 1.0),
            input_dimension=2,
            font={
                "font_title": {'family': 'sans-serif', 'color': 'black', 'size': 20},
                "font_axis_labels": {'family': 'sans-serif', 'color': 'blue', 'size': 15}
            }
        )
