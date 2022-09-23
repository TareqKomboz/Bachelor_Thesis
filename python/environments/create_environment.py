from environments.tf_env_abs_obs_abs_act import TfEnvAbsObsAbsAct


def create_environment(
        environment_type,
        input_dimension,
        function_name,
        objective_function,
        number_free_parameters,
        start_point,
        batch_size,
        episode_length):
    if environment_type == "absolute":
        return TfEnvAbsObsAbsAct(
            input_dimension=input_dimension,
            name=function_name,
            objective_function=objective_function,
            number_free_parameters=number_free_parameters,
            starting_position=start_point,
            batch_size=batch_size,
            episode_length=episode_length
        )
