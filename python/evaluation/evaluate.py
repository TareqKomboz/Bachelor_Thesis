import logging
import os
import time


import tensorflow as tf
from tf_agents.utils import common

from agents.create_agent import create_agent
from objective_functions.tf_objective_functions import FUNCTIONS

from evaluation.evaluation_driver import EvaluationDriver

def evaluate(agent_name,
             num_observations,
             plot_dir,
             environment_type):
    start_time = time.time_ns()

    policy_checkpoint_dir = os.path.join(plot_dir, "checkpoint", "policy")

    if not os.path.isdir(policy_checkpoint_dir):
        raise FileNotFoundError("No Checkpoints available at {}, evaluation is aborting".format(policy_checkpoint_dir))

    global_step = tf.compat.v1.train.get_or_create_global_step()

    eval_driver = EvaluationDriver(plot_dir, num_observations, environment_type)

    env = eval_driver.envs[0]

    agent = create_agent(agent_name,
                         env.observation_spec(),
                         env.action_spec(),
                         env.time_step_spec(),
                         step_counter=global_step)

    policy_checkpointer = common.Checkpointer(
        ckpt_dir=policy_checkpoint_dir,
        policy=agent.policy,
        global_step=global_step)

    load_status = policy_checkpointer.initialize_or_restore().expect_partial()
    load_status.assert_consumed()

    logging.info("Checkpoint at step {} loaded, commencing evaluation, this might take a while..."
                 .format(global_step.numpy()))
    performances = eval_driver.run(agent.policy, global_step.numpy())
    avg_performance = tf.reduce_sum(performances) / len(performances)

    perf_str = ""
    for label, performance in zip(FUNCTIONS.keys(), performances):
        perf_str += "{}={:.2f}, ".format(label, performance)
    logging.info("{}: performances by function: {}".format(global_step.numpy(), perf_str))

    return avg_performance, (time.time_ns() - start_time) * 1e-9

