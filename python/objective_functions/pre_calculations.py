from tf_objective_functions import FUNCTIONS
from evaluation.evaluation_utils import build_evaluation_parameters

def calc_max(objective_function, function_name, input_dimension, number_free_parameters, start_point):
    pass


if __name__ == "__main__":
    for input_dimension in [4, 6, 8, 10]:
        for number_free_parameters in [1, (input_dimension / 2), (input_dimension - 1)]:
            for function_name in FUNCTIONS.keys():
                start_points = build_evaluation_parameters(n_start_pos=4, input_dimension=input_dimension)
                for start_point in start_points:
                    calc_max(
                        objective_function=FUNCTIONS[function_name],
                        function_name=function_name,
                        input_dimension=input_dimension,
                        number_free_parameters=number_free_parameters,
                        start_point=start_point
                    )