from environments.tf_env_abs_obs_abs_act import TfEnvAbsObsAbsAct


def create_environment(
        function_names,
        objective_functions,
        start_point,
        batch_size):
    return TfEnvAbsObsAbsAct(
        name=function_names,
        objective_functions=objective_functions,
        starting_position=start_point,
        batch_size=batch_size
    )
