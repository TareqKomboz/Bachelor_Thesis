import argparse
import os
import sys
import time
import shutil

import gin
import logging

from common.build_run_id import get_run_id
from common.utils import is_valid_filename
from definitons import RUNS_DIR
from evaluation.evaluate import evaluate
from training.train import train
from objective_functions.tf_objective_functions import FUNCTIONS


@gin.configurable
def main(arguments, environment_type, agent_name, input_dimension, function_name, number_free_parameters, episode_length):
    run_id = get_run_id(arguments.configfile)

    run_dir = os.path.join(
        RUNS_DIR,
        "agent_name_{}".format(agent_name),
        "input_dimension_{}".format(input_dimension),
        "number_free_parameters_{}".format(number_free_parameters),
        function_name,
        run_id
    )

    logfile = os.path.join(run_dir, "run.log")
    if arguments.evaluate:
        logfile = os.path.join(run_dir, "eval.log")

    configfile = os.path.join(run_dir, "config.gin")
    if not os.path.isdir(run_dir):
        os.makedirs(run_dir)

    if not os.path.isfile(configfile):
        shutil.copyfile(arguments.configfile, configfile)
    elif not os.path.samefile(arguments.configfile, configfile):
        shutil.copyfile(configfile, os.path.join(run_dir, "old_config.gin"))
        shutil.copyfile(arguments.configfile, configfile)

    logging_level = logging.DEBUG if arguments.debug else logging.INFO
    logging.basicConfig(level=logging_level, handlers=[logging.FileHandler(logfile), logging.StreamHandler(sys.stdout)])

    logging.info("Loaded contents of configfile {}".format(arguments.configfile))
    logging.info("Starting run on {} environment with {} agent on {}d-{} function with {} free and {} opt parameters. run_id {}".format(
        environment_type,
        agent_name,
        input_dimension,
        function_name,
        number_free_parameters,
        (input_dimension - number_free_parameters),
        run_id
    ))
    if not arguments.evaluate:
        average_final_objective_function_value_over_batch, duration = train(
            run_dir=run_dir,
            environment_type=environment_type,
            agent_name=agent_name,
            input_dimension=input_dimension,
            function_name=function_name,
            number_free_parameters=number_free_parameters,
            episode_length=episode_length
        )
        logging.info("{}-{}d-{}free-{}opt-{} - training finished in {}, final performance = {:.2f}".format(
            run_id,
            input_dimension,
            number_free_parameters,
            (input_dimension - number_free_parameters),
            function_name,
            time.strftime('%H:%M:%S', time.gmtime(duration)),
            average_final_objective_function_value_over_batch  # final performance
        ))
    else:
        logging.info("Skipping training, evaluation only")
        final_performance, duration = evaluate(
            plot_dir=run_dir,
            environment_type=environment_type,
            agent_name=agent_name,
            input_dimension=input_dimension,
            function_name=function_name,
            number_free_parameters=number_free_parameters,
            episode_length=episode_length
        )

        logging.info("{}-{}d-{}free-{}opt-{} - evaluation finished in {}, final performance = {:.2f}".format(
            run_id,
            input_dimension,
            number_free_parameters,
            (input_dimension - number_free_parameters),
            function_name,
            time.strftime('%H:%M:%S', time.gmtime(duration)),
            final_performance
        ))


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser(description="gets gin config file")
    arg_parser.add_argument("-c",
                            "--config",
                            dest="configfile",
                            help="provide gin config file path relative to root directory",
                            type=lambda x: is_valid_filename(arg_parser, x),
                            default="default.gin")
    arg_parser.add_argument("-d",
                            "--debug",
                            dest="debug",
                            help="sets the logging verbosity to debug",
                            type=bool,
                            default=False)
    arg_parser.add_argument("-e",
                            "--evaluate",
                            dest="evaluate",
                            help="if this flag is set, the agent will be only evaluated and not trained",
                            type=bool,
                            default=False)
    args = arg_parser.parse_args()
    gin.parse_config_file(args.configfile)
    main(arguments=args)  # for debugging

    # for agent_name in ["ppo", "rnn_ppo"]:
    # for function_name in ["Zakharov"]:
    #     for input_dimension in [2, 4, 6, 8, 10]:
    #         if input_dimension == 2:
    #             main(
    #                 arguments=args,
    #                 input_dimension=input_dimension,
    #                 number_free_parameters=1,
    #                 function_name=function_name
    #             )
    #         else:
    #             for number_free_parameters in [1, int(input_dimension/2), input_dimension-1]:
    #                 main(
    #                     arguments=args,
    #                     input_dimension=input_dimension,
    #                     number_free_parameters=number_free_parameters,
    #                     function_name=function_name
    #                 )
