from environments.tf_env_abs_obs_abs_act import TfEnvAbsObsAbsAct


def create_environment(
        function_name,
        objective_function,
        start_point,
        batch_size):
    return TfEnvAbsObsAbsAct(
        name=function_name,
        objective_function=objective_function,
        starting_position=start_point,
        batch_size=batch_size
    )
