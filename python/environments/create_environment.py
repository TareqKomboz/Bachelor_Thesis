from environments.tf_env_2d_abs_obs_abs_act import TfEnv2DAbsObsAbsAct


def create_environment(type,
                       function_names,
                       obj_fcts,
                       start_point,
                       episode_length,
                       num_observations,
                       batch_size):
    if type == 'absolute':
        return TfEnv2DAbsObsAbsAct(function_names,
                                   obj_fcts,
                                   start_point,
                                   episode_length,
                                   num_observations,
                                   batch_size)
    else:
        return TfEnv2DRelObsRelAct(function_names,
                                   obj_fcts,
                                   start_point,
                                   episode_length,
                                   num_observations,
                                   batch_size)