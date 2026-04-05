import os
import gin
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
import tensorflow as tf

from objective_functions.tf_objective_functions import FUNCTIONS
from definitions import ROOT_DIR

# Ensure a directory for output exists
VIS_OUT_DIR = os.path.join(ROOT_DIR, "visualizations")
os.makedirs(VIS_OUT_DIR, exist_ok=True)


@gin.configurable()
def visualize_function_surface(objective_function, objective_function_name, parameter_bounds=(-1.0, 1.0), input_dimension=2, factor=1.0):
    """Visualizes the 2D or 3D surface of an objective function."""
    r_min, r_max = parameter_bounds
    
    font_title = {'family': 'sans-serif', 'color': 'black', 'size': 20}
    font_axis = {'family': 'sans-serif', 'color': 'blue', 'size': 15}

    if input_dimension == 1:
        x_axis = np.linspace(r_min, r_max, 1000)
        output = objective_function(x=np.array([x_axis]))

        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        plt.plot(x_axis, output, 'r')
        plt.title(f"{objective_function_name} Function", fontdict=font_title)
        plt.xlabel("$x$", fontdict=font_axis)
        plt.ylabel("$f(x)$", fontdict=font_axis)
        plt.savefig(os.path.join(VIS_OUT_DIR, f"{objective_function_name}_1D.png"))
        plt.close()

    elif input_dimension == 2:
        x_range = np.linspace(r_min, r_max, 100)
        y_range = np.linspace(r_min, r_max, 100)
        x_mesh, y_mesh = np.meshgrid(x_range, y_range)
        
        # Reshape for objective function
        coords = np.stack([x_mesh.ravel(), y_mesh.ravel()], axis=0)
        results = objective_function(x=coords).numpy()
        z_mesh = results.reshape(x_mesh.shape)

        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        surf = ax.plot_surface(X=factor * x_mesh, Y=factor * y_mesh, Z=z_mesh, cmap=cm.Blues_r)

        clean_name = objective_function_name.replace("_", " ").capitalize()
        plt.title(f"{clean_name} Function", fontdict=font_title)
        ax.set_xlabel("x", fontdict=font_axis)
        ax.set_ylabel("y", fontdict=font_axis)
        
        plt.savefig(os.path.join(VIS_OUT_DIR, f"{objective_function_name}_3D.png"))
        plt.close()


def visualize_all_benchmarks():
    """Generates 3D visualizations for all supported benchmark functions."""
    factors = {
        "Ackley": 20,
        "Griewank": 600,
        "Levy": 10,
        "Rastrigin": 5,
        "Rosenbrock": 2,
        "Sphere": 5,
        "Styblinski_tang": 5,
        "Zakharov": 5
    }
    
    for key, func in FUNCTIONS.items():
        print(f"Visualizing {key}...")
        visualize_function_surface(
            objective_function=func,
            objective_function_name=key,
            factor=factors.get(key, 1.0)
        )


if __name__ == "__main__":
    visualize_all_benchmarks()
