# Parameter-Dependent Learning to Optimize (L2O) via Reinforcement Learning

This project implements a professional and modular framework for **Learning to Optimize (L2O)** using Reinforcement Learning. Originally developed for my Bachelor's Thesis, it has been refactored for a high-quality PhD application showcase.

The framework trains RL agents (REINFORCE and PPO) to act as optimizers for high-dimensional mathematical functions. A key focus is **Parameter-Dependent Optimization**, where the agent learns to optimize a subset of parameters while being conditioned on fixed "free parameters" (context).

## ✨ Key Features
- **Custom RL Environments:** High-performance TensorFlow-based environments specifically designed for optimization benchmarks.
- **Benchmark Suite:** Support for standard functions: Ackley, Griewank, Levy, Rastrigin, Rosenbrock, Sphere, Styblinski-Tang, and Zakharov.
- **Modern Package Structure:** Fully integrated `l2o` package following professional Python conventions.
- **Config-Driven Experiments:** Reproducible runs managed via `gin-config`.

## 📂 Project Structure

```text
Bachelor_Thesis/
├── l2o/                # Core Package (Library & Tools)
│   ├── agents/         # REINFORCE & PPO Architectures
│   ├── environments/   # Custom Optimization Environments
│   ├── objective_functions/ # Benchmark implementations
│   ├── training/       # Training drivers and batch scripts
│   ├── evaluation/     # Evaluation, plotting, and analysis
│   └── utils/          # Project management and data utilities
├── configs/            # Experiment configurations (.gin)
├── main.py             # Main entry point (Wrapper)
├── requirements.txt    # Project dependencies
└── README.md
```

## 🚀 Getting Started

### 1. Installation
The project requires Python 3.x and TensorFlow.
```bash
pip install -r requirements.txt
```

### 2. Training an Agent
Run the main entry point with a configuration file:
```bash
python main.py -c configs/default.gin
```
Or use the package execution:
```bash
python -m l2o -c configs/default.gin
```

### 3. Running Tools
Utilities are integrated into the package. For example, to evaluate all runs:
```bash
python -m l2o.evaluation.evaluate_all -a -t 8
```

---
*Developed as part of my Bachelor's Thesis. Refactored and modernized in 2026.*
