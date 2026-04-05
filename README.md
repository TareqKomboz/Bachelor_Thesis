# Parameter-Dependent Learning to Optimize (L2O) via Reinforcement Learning

This repository implements a framework for **Learning to Optimize (L2O)** using Reinforcement Learning. The project explores training RL agents to act as optimizers that can adapt to specific characteristics of function families, particularly in high-dimensional spaces with "free parameters" (fixed context) and "optimization parameters" (variables to be adjusted).

## Core Components

*   **l2o Package:** The core library containing:
    *   `agents/`: Implementation of REINFORCE and PPO agents using `tf_agents`.
    *   `environments/`: Custom TensorFlow-based environments for optimization.
    *   `objective_functions/`: Benchmark functions (Ackley, Rosenbrock, etc.) in TensorFlow.
    *   `evaluation/` & `training/`: Drivers for the RL lifecycle.
    *   `analysis/`: Scripts for MSE calculations and visualizations.
*   **Configuration:** Experiment settings managed via `gin-config` in the `configs/` directory.

## How to Run

### 1. Training
To train an agent, use the `main.py` entry point with a `.gin` configuration file.
```bash
python main.py -c configs/default.gin
```
Logs and checkpoints are saved in the `runs/` directory.

### 2. Evaluation
To evaluate a trained policy, use the `-e True` flag:
```bash
python main.py -c runs/.../config.gin -e True
```

### 3. Utility Scripts
The `scripts/` directory contains tools for batch processing and management:
*   **Evaluate all runs:** `python scripts/evaluation/evaluate_all.py -a -t <threads>`
*   **Generate summary:** `python scripts/evaluation/build_summary_table.py`

## Requirements
Ensure you have the following installed:
*   TensorFlow & TF-Agents
*   Gin-config
*   NumPy, Pandas, Matplotlib, Natsort
(See `requirements.txt` for details)
