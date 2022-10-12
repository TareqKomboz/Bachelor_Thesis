import gin


@gin.configurable(allowlist=['number_evaluation_episodes'])
def compute_average_objective_function_value_and_return(
        evaluation_environment,
        evaluation_policy,
        number_evaluation_episodes,
        objective_function):
    final_objective_function_value_sum = 0.0
    total_return = 0.0
    for _ in range(number_evaluation_episodes):
        time_step = evaluation_environment.reset()
        episode_return = 0.0
        current_action = None
        policy_state = evaluation_policy.get_initial_state(evaluation_environment.batch_size)

        while not time_step.is_last():
            action_step = evaluation_policy.action(time_step, policy_state=policy_state)
            policy_state = action_step.state
            current_action = action_step.action
            time_step = evaluation_environment.step(current_action)
            episode_return += time_step.reward.numpy()[0]

        current_final_objective_function_value = objective_function(x=current_action.numpy()[0])
        final_objective_function_value_sum += current_final_objective_function_value
        total_return += episode_return

    average_final_objective_function_value = final_objective_function_value_sum / number_evaluation_episodes
    average_episode_return = total_return / number_evaluation_episodes

    return average_final_objective_function_value, average_episode_return
