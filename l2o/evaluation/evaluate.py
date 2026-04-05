import logging
import os
import time


import tensorflow as tf
from tf_agents.utils import common

from l2o.agents.create_agent import create_agent
from l2o.objective_functions.tf_objective_functions import FUNCTIONS

from l2o.evaluation.evaluation_driver import EvaluationDriver


def evaluate(plot_dir, environment_type, agent_name, input_dimension, function_name, number_free_parameters, episode_length):
    start_time = time.time_ns()

    policy_checkpoint_dir = os.path.join(plot_dir, "checkpoint", "policy")

    if not os.path.isdir(policy_checkpoint_dir):
        raise FileNotFoundError("No Checkpoints available at {}, evaluation is aborting".format(policy_checkpoint_dir))

    global_step = tf.compat.v1.train.get_or_create_global_step()

    evaluation_driver = EvaluationDriver(
        run_dir=plot_dir,
        environment_type=environment_type,
        input_dimension=input_dimension,
        function_name=function_name,
        number_free_parameters=number_free_parameters,
        episode_length=episode_length
    )

    environment = evaluation_driver.environment

    agent = create_agent(
        name=agent_name,
        obs_spec=environment.observation_spec(),
        act_spec=environment.action_spec(),
        ts_spec=environment.time_step_spec(),
        step_counter=global_step
    )

    policy_checkpointer = common.Checkpointer(
        ckpt_dir=policy_checkpoint_dir,
        policy=agent.policy,
        global_step=global_step
    )

    load_status = policy_checkpointer.initialize_or_restore().expect_partial()
    load_status.assert_consumed()

    logging.info("Checkpoint at step {} loaded, commencing evaluation, this might take a while..."
                 .format(global_step.numpy()))
    performances = evaluation_driver.run(agent.policy, global_step.numpy())
    average_performance = tf.reduce_sum(performances) / len(performances)

    perf_str = ""
    for label, performance in zip(FUNCTIONS.keys(), performances):
        perf_str += "{}={:.2f}, ".format(label, performance)
    logging.info("{}: performances by function: {}".format(global_step.numpy(), perf_str))

    return average_performance, (time.time_ns() - start_time) * 1e-9
