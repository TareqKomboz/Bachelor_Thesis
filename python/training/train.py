import os
import time

import gin
import tensorflow as tf
import logging

from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.utils import common

from agents.create_agent import create_agent
from objective_functions.tf_objective_functions import FUNCTIONS
from common.utils import format_function_names
from environments.create_environment import create_environment
from evaluation.evaluation_driver import EvaluationDriver
from evaluation.evaluation_utils import plot_returns_and_losses
from training.training_driver import TrainingDriver

from numpy import ones


@gin.configurable
def train(run_dir,

          # gin parameter
          function_names,
          agent_name,
          # environment parameters
          batch_size,
          input_dimension,
          randomize_start,
          randomization_interval,
          replay_buffer_capacity,

          # training parameters
          number_iterations,
          log_interval,
          eval_interval,
          quick_eval_interval,
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

    objective_functions = [FUNCTIONS[function_name][0] for function_name in function_names]

    train_env = create_environment(
        function_names=format_function_names(function_names),
        objective_functions=objective_functions,
        start_point=start_point,
        batch_size=batch_size
    )

    # Create evaluation and plotting environments
    eval_driver = EvaluationDriver(run_dir=run_dir)

    agent = create_agent(
        agent_name,
        train_env.observation_spec(),
        train_env.action_spec(),
        train_env.time_step_spec(),
        step_counter=global_step)

    replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
        agent.collect_data_spec,
        batch_size=train_env.batch_size,
        max_length=replay_buffer_capacity
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
    load_status = policy_checkpointer.initialize_or_restore().expect_partial()
    rb_checkpointer.initialize_or_restore().expect_partial()
    try:
        load_status.assert_consumed()
        checkpoint_loaded = True
    except AssertionError:
        checkpoint_loaded = False
    logging.info("Setup completed in {:.2f}s".format((time.time_ns() - start_time) * 1e-9))

    # training loop
    onpolicy = agent_name != "sac"
    driver = TrainingDriver(agent, train_env, replay_buffer, replay_observer, number_iterations, onpolicy)

    logging.info("Starting training loop, initial graph construction might take a bit")
    log_timestamp = time.time_ns()
    latest_eval_step = 0
    latest_quick_eval_step = 0
    for i in range(number_iterations):

        timestamp = time.strftime('%H:%M:%S', time.gmtime((time.time_ns() - start_time) * 1e-9))
        print("{}: {} ({}) train loops completed, {}"
              .format(global_step.numpy(), (global_step.numpy() % log_interval), log_interval, timestamp), end="")

        driver.train_step()
        if global_step % quick_eval_interval == 0:
            driver.quick_eval()

        print("", end="\r")

        if global_step % randomization_interval == 0:
            if randomize_start:
                start_point = tf.random.uniform(
                    shape=(batch_size, input_dimension),
                    minval=tuple((-1 * ones((input_dimension,), dtype=int)).tolist()),
                    maxval=tuple((ones((input_dimension,), dtype=int)).tolist()),
                    dtype=tf.float32
                )
                train_env.set_starting_positions(start_point)

        if global_step % log_interval == 0:
            rewards, losses, performances = driver.get_summary()

            final_performance = tf.reduce_mean(performances[driver.step_eval - quick_eval_interval:driver.step_eval, 0])
            avg_performance = tf.reduce_mean(performances[driver.step_eval - quick_eval_interval:driver.step_eval, 1])

            avg_reward = tf.reduce_mean(rewards[driver.step - log_interval:driver.step])
            avg_loss = tf.reduce_mean(losses[driver.step - log_interval:driver.step])

            timestamp = time.gmtime((time.time_ns() - start_time) * 1e-9)
            logging.info("{}: {} train loops completed in {:.2f}s, "
                         "train reward={:.2f}, "
                         "loss={:.2f}, "
                         "final perf={:.2f}, avg perf={:.2f}, "
                         "{}"
                         .format(global_step.numpy(),
                                 log_interval,
                                 (time.time_ns() - log_timestamp) * 1e-9,
                                 avg_reward,
                                 avg_loss,
                                 final_performance,
                                 avg_performance,
                                 time.strftime('%H:%M:%S', timestamp)))
            log_timestamp = time.time_ns()

        is_last_iteration = i == (number_iterations - 1)
        
        if global_step % checkpoint_interval == 0 or is_last_iteration:
            train_checkpointer.save(global_step=global_step)
            policy_checkpointer.save(global_step=global_step)
            rb_checkpointer.save(global_step=global_step)

        if global_step % eval_interval == 0 or is_last_iteration:
            performances = eval_driver.run(agent.policy, global_step.numpy())

            avg_performance = tf.reduce_mean(performances)

            returns, train_losses, train_performances = driver.get_summary()
            plot_returns_and_losses(returns.numpy()[latest_eval_step:],
                                    train_losses.numpy()[latest_eval_step:],
                                    train_performances.numpy()[latest_quick_eval_step:],
                                    plot_dir, agent_name, function_names, quick_eval_interval)

            timestamp = time.gmtime((time.time_ns() - start_time) * 1e-9)
            logging.info("{}: Evaluation completed in {:.2f}s, average performance = {:.2f}, {}"
                         .format(global_step.numpy(),
                                 (time.time_ns() - log_timestamp) * 1e-9,
                                 avg_performance,
                                 time.strftime('%H:%M:%S', timestamp)))

            perf_str = ""
            for label, performance in zip(FUNCTIONS.keys(), performances):
                perf_str += "{}={:.2f}, ".format(label, performance)
            logging.info("{}: performances by function: {}".format(global_step.numpy(), perf_str))
            log_timestamp = time.time_ns()
            latest_eval_step = driver.step.numpy()
            latest_quick_eval_step = driver.step_eval.numpy()

    return avg_performance, ((time.time_ns() - start_time) * 1e-9)
