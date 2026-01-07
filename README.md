# Parameter-Dependent Self-Learning Optimization

This repository contains our implementation of our testbench. 
We provide a short guide to using our code for training and evaluation here.
# Training
For training, create a `config.gin` file containing all hyperparameters located in the root directory.
The `default.gin` file contains the default values as presented in our work.
Start training with:
```
python python/main.py -c config.gin
```
If no config file is provided, the `default.gin` file is used.
The training process is logged to the console, as well as to the `runs` folder located in the root directory.
The structure of the `runs` folder is: `runs/agent_name/function_name/run_id`. Therefore, every unique run has its folder. 
`agent_name` and `function_name` are given by the hyperparameters.
`run_id` is built based on hyperparameters different from the defaults. 
For details we refer to `python/common/build_run_id.py`

# Evaluation
For evaluation, you need a checkpoint of a policy in the corresponding run folder as such: `runs/agent_name/function_name/run_id/checkpoints/policy`
Further, a `config.gin` file with the same values as the policy checkpoint.
Then run:
```
python python/main.py -c config.gin -e True
```
The results are output to a folder in the run folder as such `run_folder/Step_global_step` where global_step is the amount of training iterations the policy has already been trained on
If the SQL saving is enabled, all performances over time are put into an SQL table named `runs`.

To evaluate all runs in the `runs` folder in parallel, run 
```
python python/scripts/evaluation/evaluate_all.py -a -t number_of_threads
```
To create a summary of all evaluations
```
python python/scripts/evaluation/build_summary_table.py
```
The output is in the `runs` folder, if SQL saving is set up, the summary is also in a table named `performance`
