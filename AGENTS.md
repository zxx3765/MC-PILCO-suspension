# AGENTS.md

This file provides guidance to Codex (Codex.ai/code) when working with code in this repository.

## Project Overview

MC-PILCO (Monte Carlo Probabilistic Inference for Learning and COntrol) is a Model-based Reinforcement Learning algorithm for modeling and control of dynamical systems. It uses Gaussian Processes (GPs) to model system dynamics and Monte Carlo methods for policy gradient estimation during optimization.

**Key variant**: MC-PILCO-4PMS extends the algorithm to Partially Measurable Systems, modeling both the measurement system and state estimators during policy optimization.

## Environment Setup

Create the conda environment:
```bash
conda env create --file environment.yaml
conda activate mc-pilco
```

**Dependencies**: PyTorch 1.4+, NumPy, Matplotlib, Pickle, Argparse. Optional: MuJoCo 2.00, MuJoCo-Py, Gym (for MuJoCo environments).

## Running Tests

All test scripts are in the root directory and should be run from there:

```bash
# Standard MC-PILCO tests
python test_mcpilco_cartpole.py                    # Cartpole with SE+polynomial kernel
python test_mcpilco_cartpole_rbf_ker.py            # Cartpole with SE kernel only
python test_mcpilco_cartpole_multi_init.py         # Cartpole with multiple initial positions

# MC-PILCO-4PMS (Partially Measurable Systems)
python test_mcpilco4pms_cartpole.py                # Cartpole with sensors and state estimation

# MuJoCo environments (requires MuJoCo installation)
python test_mcpilco_cartpole_mujoco.py             # Cartpole in MuJoCo
python test_mcpilco_ur5_mujoco.py                  # UR5 robot arm joint-space controller

# Apply learned policies
python apply_mcpilco_policy.py                     # Apply MC-PILCO policy
python apply_mcpilco_policy_on_model.py            # Apply policy on learned model
python apply_mcpilco4pms_policy.py                 # Apply MC-PILCO-4PMS policy
python apply_mcpilco4pms_policy_on_model.py        # Apply 4PMS policy on model
```

**Random seed**: Tests accept `-seed` argument (e.g., `python test_mcpilco_cartpole.py -seed 42`).

**Repeat tests**: Use `repeat_test.py` for Monte Carlo simulations with statistical analysis.

## Code Architecture

### Core Components

**`policy_learning/`** - Policy optimization and cost functions
- `MC_PILCO.py`: Main MC-PILCO class implementing the reinforcement learning loop
  - `reinforce()`: Main training loop with exploration and policy optimization phases
  - Manages data collection, model learning, and policy updates
- `MC_PILCO_mujoco_envs.py`: MC-PILCO variant for MuJoCo environments
- `Policy.py`: Policy classes (Random_exploration, RBF_Policy, Linear_Policy, etc.)
- `Cost_function.py`: Cost function implementations for different tasks

**`model_learning/`** - Gaussian Process model learning
- `Model_learning.py`: GP-based system dynamics learning
  - Supports exact GP inference and approximations (SOR, SOD)
  - `Speed_Model_learning_*` classes: Specialized for velocity-based state representations
  - Handles angle wrapping and state transformations

**`simulation_class/`** - System simulation
- `model.py`: ODE-based system simulation wrapper
- `model_mujoco.py`: MuJoCo environment wrapper
- `ode_systems.py`: ODE definitions for simulated systems (cartpole, etc.)

**`gpr_lib/`** - Gaussian Process Regression library (courtesy of Alberto Dalla Libera)
- `GP_prior/`: GP prior implementations (GP_prior.py, Sparse_GP.py, Stationary_GP.py)
- `Likelihood/`: Likelihood functions (Gaussian_likelihood.py)
- `Utils/`: Covariance functions and scaling utilities

**`envs/`** - Environment definitions
- `cartpole_swingup.py`: Cartpole environment for Gym/MuJoCo
- `ur5.py`: UR5 robot environment
- `assets/`: MuJoCo XML model files

### Key Workflow

1. **Exploration phase**: Random policy collects initial data from the system
2. **Model learning**: GPs learn system dynamics from collected data
3. **Policy optimization**: Monte Carlo gradient estimation optimizes policy parameters
4. **Control phase**: Optimized policy is applied and new data is collected
5. **Iteration**: Steps 2-4 repeat for multiple trials

### Important Implementation Details

- **State representation**: Many models use velocity-based representations where GPs predict state derivatives rather than next states
- **Angle handling**: Special treatment for angular states (wrapping, sine/cosine encoding)
- **GP approximations**: SOD (Subset of Data) and SOR (Subset of Regressors) for computational efficiency
- **Dropout**: Used during policy optimization for robustness
- **Cost shaping**: Critical for effective Monte Carlo policy gradient estimation

## Code Style

- **Formatting**: Black with 120 character line length
- **Import sorting**: isort with Black profile
- **Linting**: flake8 (warnings only, doesn't fail pre-commit)
- **Pre-commit hooks**: Run `pre-commit install` to enable automatic formatting

Format code:
```bash
black --line-length=120 <file>
isort --profile black --line-length 120 <file>
```

## Logging and Results

- Test scripts save results to `results_tmp/` by default
- Use `log_plot_*.py` scripts to visualize results:
  - `log_plot_cartpole.py`: Plot cartpole results
  - `log_plot_cartpole_mujoco.py`: Plot MuJoCo cartpole results
  - `log_plot_ur5.py`: Plot UR5 results

## License

AGPL-3.0-or-later. All files must include MERL copyright header.
