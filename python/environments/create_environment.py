from environments.tf_env_abs_obs_abs_act import TfEnvAbsObsAbsAct


def create_environment(
        type,
        function_names,
        objective_functions,
        start_point,
        episode_length,
        number_observations,
        batch_size,
        input_dimension,
        number_optimization_parameters):
    if type == 'absolute':
        return TfEnvAbsObsAbsAct(
            function_names,
            objective_functions,
            start_point,
            episode_length,
            number_observations,
            batch_size,
            input_dimension,
            number_optimization_parameters
        )
