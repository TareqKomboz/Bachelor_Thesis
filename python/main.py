import argparse
import os
import sys
import time
import shutil

import gin
import logging

from common.build_run_id import get_run_id
from objective_functions.tf_objective_functions import FUNCTIONS
from common.utils import is_valid_filename
from definitons import RUNS_DIR
from evaluation.evaluate import evaluate
from training.train import train


@gin.configurable
def main(arguments, agent_name, function_name, environment_type):
    run_id = get_run_id(arguments.configfile)

    run_dir = os.path.join(RUNS_DIR, agent_name, function_name, run_id)

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
    logging.basicConfig(level=logging_level,
                        handlers=[
                            logging.FileHandler(logfile),
                            logging.StreamHandler(sys.stdout)
                        ])

    logging.info("Loaded contents of configfile {}".format(arguments.configfile))
    logging.info("Starting run with {} agent, {} function and id {}"
                 .format(agent_name, function_name, run_id))
    if not arguments.evaluate:
        final_performance, duration = train(run_dir=run_dir)
        logging.info("{}-{}-{} training finished in {}, final performance = {:.2f}"
                     .format(run_id,
                             agent_name,
                             function_name,
                             time.strftime('%H:%M:%S', time.gmtime(duration)),
                             final_performance))
    else:
        logging.info("Skipping training, evaluation only")
        final_performance, duration = evaluate(agent_name=agent_name, plot_dir=run_dir)

        logging.info("{}-{}-{} evaluation finished, final performance = {:.2f}, {}"
                     .format(run_id,
                             agent_name,
                             function_name,
                             final_performance,
                             time.strftime('%H:%M:%S', time.gmtime(duration))
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
                            # action=argparse.BooleanOptionalAction,
                            default=False)
    args = arg_parser.parse_args()
    gin.parse_config_file(args.configfile)
    main(arguments=args)
