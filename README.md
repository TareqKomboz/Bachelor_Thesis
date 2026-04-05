# Parameter-Dependent Learning to Optimize (L2O) via Reinforcement Learning

This repository implements a framework for **Learning to Optimize (L2O)** using Reinforcement Learning. The project explores training RL agents to act as optimizers that can adapt to specific characteristics of function families, particularly in high-dimensional spaces with "free parameters" (fixed context) and "optimization parameters" (variables to be adjusted).

## Core Components

*   **Agents:** Implementation of REINFORCE and PPO agents using `tf_agents`.
*   **Environments:** Custom TensorFlow-based environments (`TFEnvironment`) designed for high-dimensional optimization tasks.
*   **Benchmark Functions:** Support for a variety of standard optimization benchmarks, including:
    *   Ackley, Griewank, Levy, Rastrigin, Rosenbrock, Sphere, Styblinski-Tang, and Zakharov functions.
*   **Configuration:** Extensive use of `gin-config` to ensure experiment reproducibility and modular parameter management.

## Project Structure

*   `python/main.py`: Entry point for both training and evaluation.
*   `python/agents/`: Agent creation and architecture definitions.
*   `python/environments/`: Custom RL environments for optimization.
*   `python/objective_functions/`: Implementations of the benchmark functions in TensorFlow.
*   `python/scripts/`: Utility scripts for batch evaluation, plotting, and data management.

## How to Run

### 1. Training
To train an agent, use a `.gin` configuration file. The `default.gin` in the root directory provides a base configuration.
```bash
python python/main.py -c default.gin
```
Training logs and checkpoints are saved in the `runs/` directory following the structure:
`runs/agent_name_.../input_dimension_.../number_free_parameters_.../function_name/run_id/`

### 2. Evaluation
To evaluate a trained policy, use the `-e True` flag. The configuration file passed should match the one used during training.
```bash
python python/main.py -c runs/.../config.gin -e True
```
Evaluation results are output to a `Step_global_step` folder within the specific run directory.

### 3. Batch Evaluation & Plotting
*   **Evaluate all runs in parallel:**
    ```bash
    python python/scripts/evaluation/evaluate_all.py -a -t <number_of_threads>
    ```
*   **Generate summary table:**
    ```bash
    python python/scripts/evaluation/build_summary_table.py
    ```
*   **Plotting:** Specialized plotting scripts are available in `python/scripts/plotting/` to visualize convergence (e.g., `plot_performance_from_runs_table.py`).

## Requirements
The project requires Python 3.x and the following libraries:
*   TensorFlow
*   TF-Agents
*   Gin-config
*   NumPy
*   Pandas
*   Matplotlib
*   Natsort
