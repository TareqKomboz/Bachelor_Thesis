import os
import time

import gin
import tensorflow as tf
import logging

from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.utils import common

from l2o.agents.create_agent import create_agent
from l2o.objective_functions.tf_objective_functions import FUNCTIONS
from l2o.environments.create_environment import create_environment
from l2o.evaluation.evaluation_driver import EvaluationDriver
from l2o.evaluation.evaluation_utils import plot_returns_and_losses
from l2o.training.training_driver import TrainingDriver

from l2o.common.utils import save_data

from numpy import ones, asarray


@gin.configurable
def train(
        run_dir,
        environment_type,
        agent_name,
        input_dimension,
        function_name,
        number_free_parameters,
        episode_length,

        # environment parameters
        batch_size,
        randomize_start,

        # training parameters
        number_training_iterations,
        log_interval,
        evaluation_interval,
        quick_evaluation_interval,
        checkpoint_interval):
    start_time = time.time_ns()
    logging.info("Starting training setup")

    checkpoint_dir = os.path.join(run_dir, "checkpoint")
    plot_dir = run_dir

    if not os.path.isdir(plot_dir):
        os.makedirs(plot_dir)

    # tf.config.threading.set_intra_op_parallelism_threads(4)
    # tf.config.threading.set_inter_op_parallelism_threads(1)

    global_step = tf.compat.v1.train.get_or_create_global_step()

    # create train environment
    if randomize_start:
        start_point = tf.random.uniform(
            shape=(batch_size, input_dimension),
            minval=tuple((-1 * ones((input_dimension,), dtype=int)).tolist()),
            maxval=tuple((ones((input_dimension,), dtype=int)).tolist()),
            dtype=tf.float32
        )
    else:
        start_point = tf.constant(0.8, shape=(batch_size, input_dimension), dtype=tf.float32)

    train_environment = create_environment(
        environment_type=environment_type,
        input_dimension=input_dimension,
        function_name=function_name,
        objective_function=FUNCTIONS[function_name],
        number_free_parameters=number_free_parameters,
        start_point=start_point,
        batch_size=batch_size,
        episode_length=episode_length
    )

    # Create evaluation and plotting environments
    evaluation_driver = EvaluationDriver(
        run_dir=run_dir,
        environment_type=environment_type,
        input_dimension=input_dimension,
        function_name=function_name,
        number_free_parameters=number_free_parameters,
        episode_length=episode_length
    )

    agent = create_agent(
        name=agent_name,
        obs_spec=train_environment.observation_spec(),
        act_spec=train_environment.action_spec(),
        ts_spec=train_environment.time_step_spec(),
        step_counter=global_step
    )

    replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
        agent.collect_data_spec,
        batch_size=train_environment.batch_size,
        max_length=episode_length
    )

    replay_observer = [replay_buffer.add_batch]

    train_checkpointer = common.Checkpointer(
        ckpt_dir=checkpoint_dir,
        agent=agent,
        global_step=global_step
    )
    policy_checkpointer = common.Checkpointer(
        ckpt_dir=os.path.join(checkpoint_dir, 'policy'),
        policy=agent.policy,
        global_step=global_step
    )
    rb_checkpointer = common.Checkpointer(
        ckpt_dir=os.path.join(checkpoint_dir, 'replay_buffer'),
        max_to_keep=1,
        replay_buffer=replay_buffer
    )

    train_checkpointer.initialize_or_restore().expect_partial()
    policy_checkpointer.initialize_or_restore().expect_partial()
    rb_checkpointer.initialize_or_restore().expect_partial()

    logging.info("Setup completed in {:.2f}s".format((time.time_ns() - start_time) * 1e-9))

    # training loop
    on_policy = agent_name != "sac"
    training_driver = TrainingDriver(
        agent=agent,
        environment=train_environment,
        replay_buffer=replay_buffer,
        replay_observer=replay_observer,
        number_training_iterations=number_training_iterations,
        clear_buffer=on_policy
    )

    logging.info("Starting training loop, initial graph construction might take a bit")
    log_timestamp = time.time_ns()
    full_metric_list = []
    for i in range(number_training_iterations):

        timestamp = time.strftime('%H:%M:%S', time.gmtime((time.time_ns() - start_time) * 1e-9))
        print("{}: {} ({}) train loops completed, {}".format(
            global_step.numpy(),
            (global_step.numpy() % log_interval),
            log_interval,
            timestamp
        ), end="")

        training_driver.train_step()
        if global_step % quick_evaluation_interval == 0:
            training_driver.quick_evaluation()

        print("", end="\r")

        if randomize_start:
            start_point = tf.random.uniform(
                shape=(batch_size, input_dimension),
                minval=tuple((-1 * ones((input_dimension,), dtype=int)).tolist()),
                maxval=tuple((ones((input_dimension,), dtype=int)).tolist()),
                dtype=tf.float32
            )
            train_environment.set_starting_positions_and_free_values(start_point)

        if global_step % log_interval == 0:
            train_rewards, train_losses, evaluation_performances = training_driver.get_summary()

            average_average_final_reward_over_last_quick_evaluation_steps = \
                tf.reduce_mean(evaluation_performances[
                               training_driver.step_evaluation - quick_evaluation_interval:training_driver.step_evaluation,
                               0])
            average_average_evaluation_reward_over_last_quick_evaluation_steps = \
                tf.reduce_mean(evaluation_performances[
                               training_driver.step_evaluation - quick_evaluation_interval:training_driver.step_evaluation,
                               1])

            average_train_reward_over_last_log_interval_steps = tf.reduce_mean(
                train_rewards[training_driver.step - log_interval:training_driver.step])
            average_train_loss_over_last_log_interval_steps = tf.reduce_mean(
                train_losses[training_driver.step - log_interval:training_driver.step])

            timestamp = time.gmtime((time.time_ns() - start_time) * 1e-9)
            logging.info(
                "{}: {} train loops completed in {:.2f}s, train reward={:.2f}, loss={:.2f}, final reward={:.2f}, eval reward={:.2f}, {}".format(
                    global_step.numpy(),
                    log_interval,
                    (time.time_ns() - log_timestamp) * 1e-9,
                    average_train_reward_over_last_log_interval_steps,
                    average_train_loss_over_last_log_interval_steps,
                    average_average_final_reward_over_last_quick_evaluation_steps,
                    average_average_evaluation_reward_over_last_quick_evaluation_steps,
                    time.strftime('%H:%M:%S', timestamp))
            )
            log_timestamp = time.time_ns()

        is_last_iteration = i == (number_training_iterations - 1)

        if global_step % checkpoint_interval == 0 or is_last_iteration:
            train_checkpointer.save(global_step=global_step)
            policy_checkpointer.save(global_step=global_step)
            rb_checkpointer.save(global_step=global_step)

        if global_step % evaluation_interval == 0 or is_last_iteration:
            last_metric_list = evaluation_driver.run(agent.policy, global_step.numpy())
            full_metric_list.append(last_metric_list)

            timestamp = time.gmtime((time.time_ns() - start_time) * 1e-9)
            logging.info("{}: Evaluation completed in {:.2f}s, average performance = {:.2f}, {}".format(
                global_step.numpy(),
                (time.time_ns() - log_timestamp) * 1e-9,
                last_metric_list[4],  # average_final_objective_function_value_over_batch
                time.strftime('%H:%M:%S', timestamp))
            )

            log_timestamp = time.time_ns()

    # save/plot full_metric_list
    full_metric_array = asarray(full_metric_list)
    train_rewards, train_losses, evaluation_performances = training_driver.get_summary()

    plot_returns_and_losses(
        train_rewards=train_rewards.numpy(),
        train_losses=train_losses.numpy(),
        evaluation_performances=evaluation_performances.numpy(),
        plot_dir=plot_dir,
        input_dimension=input_dimension,
        number_free_parameters=number_free_parameters,
        function_name=function_name,
        quick_evaluation_interval=quick_evaluation_interval
    )

    csv_dir = os.path.join(run_dir, "CSVs")

    # control
    if not os.path.isdir(csv_dir):
        os.makedirs(csv_dir)

    save_data(
        csv_dir=csv_dir,
        data={
            os.path.join(csv_dir, 'not_normalized_full_metric_list.csv'): full_metric_array,
            os.path.join(csv_dir, 'not_normalized_last_metric_list.csv'): asarray(last_metric_list),

            os.path.join(csv_dir, 'not_normalized_average_reward_over_batches_and_actions.csv'): full_metric_array[:, 0],
            os.path.join(csv_dir, 'not_normalized_reward_stds_over_batches_and_actions.csv'): full_metric_array[:, 1],
            os.path.join(csv_dir, 'not_normalized_average_return_over_batch.csv'): full_metric_array[:, 2],
            os.path.join(csv_dir, 'not_normalized_return_stds_over_batch.csv'): full_metric_array[:, 3],
            os.path.join(csv_dir, 'not_normalized_average_final_objective_function_value_over_batch.csv'): full_metric_array[:, 4],
            os.path.join(csv_dir, 'not_normalized_final_objective_function_value_stds_over_batch.csv'): full_metric_array[:, 5],
            os.path.join(csv_dir, 'not_normalized_average_max_reward_of_episode_over_batches.csv'): full_metric_array[:, 6],
            os.path.join(csv_dir, 'not_normalized_max_reward_of_episode_stds_over_batches.csv'): full_metric_array[:, 7],

            os.path.join(csv_dir, 'not_normalized_train_rewards.csv'): train_rewards.numpy(),
            os.path.join(csv_dir, 'not_normalized_train_losses.csv'): train_losses.numpy(),
            os.path.join(csv_dir, 'not_normalized_evaluation_performances.csv'): evaluation_performances.numpy(),

            os.path.join(csv_dir, 'not_normalized_execution_time.csv'): asarray([((time.time_ns() - start_time) * 1e-9)])
        }
    )

    return last_metric_list[4], ((time.time_ns() - start_time) * 1e-9)
