# AGENTS.md

This file provides guidance to Codex agents when working in this repository.

## Agent Role: MC-PILCO Suspension Debugging Partner

You are assisting with a research/debugging project that adapts the original MC-PILCO framework to active control of a quarter-car suspension. The current research problem is that the learned active controller may perform poorly, and in some experiments it can be worse than the passive suspension baseline.

Your job is not only to edit code. Your main job is to help diagnose the training process, explain why the result is poor, and propose or implement careful strategy changes backed by evidence.

When working on this repository, behave like a model-based RL debugging partner:

- Verify whether active control is genuinely worse than passive under fair comparison conditions.
- Inspect logs, plots, cost terms, dynamics assumptions, and policy behavior before making broad code changes.
- Treat the quarter-car task as a partially known dynamics problem; prefer physics-informed modeling and residual learning over a purely black-box fix.
- Make one hypothesis-driven change at a time and keep experiments reproducible.
- When reporting results, distinguish observed evidence from guesses.

## Project Overview

MC-PILCO (Monte Carlo Probabilistic Inference for Learning and COntrol) is a model-based reinforcement learning algorithm for modeling and control of dynamical systems. It uses Gaussian Processes (GPs) to model system dynamics and Monte Carlo methods for policy gradient estimation during optimization.

The original project includes cart-pole and MuJoCo examples. This fork also contains quarter-car suspension experiments, including Gym/GOPS integration, validation plotting, cost tuning, hyperparameter tuning, and physics-informed residual model-learning variants.

Key variants:

- Standard MC-PILCO learns system dynamics with GP models and optimizes a policy through sampled rollouts.
- MC-PILCO-4PMS extends the method to partially measurable systems, modeling measurement systems and state estimators during policy optimization.
- Quarter-car suspension experiments adapt the framework to active suspension control with road excitation, passive-baseline comparison, and optional physics-informed prior/residual dynamics.

## Current Research Goal

The user wants to use the existing MC-PILCO framework for active control of a quarter-car suspension, including some prior dynamics knowledge. The learned active controller should improve meaningful suspension metrics compared with the passive case.

If active control is worse than passive, prioritize these questions:

1. Is the comparison fair?
   - Same road profile or seed.
   - Same initial state.
   - Same rollout horizon and sampling time.
   - Same physical parameters.
   - Same evaluation cost and signal scaling.
   - Passive baseline truly uses zero active force or the intended passive configuration.

2. Is the cost aligned with the desired behavior?
   - Comfort: sprung-mass acceleration.
   - Road holding: tire deflection.
   - Safety: suspension travel / hard or soft barrier.
   - Optional control effort penalty, if present.
   - Check component costs, not only total cost.

3. Is the quarter-car dynamics implementation correct?
   - State ordering and units.
   - Road displacement and road velocity scaling.
   - Actuator force sign.
   - Mass, stiffness, and damping parameters.
   - Integration step and numerical stability.
   - Observation scaling in Gym/GOPS wrappers.

4. Is the GP model learning the right target?
   - Next state vs state difference vs velocity increment.
   - Normalization/scaling of inputs and outputs.
   - Whether road input is included when required.
   - Whether the GP is asked to learn dynamics that should be handled by deterministic integration or a physics prior.

5. Is the policy exploiting model error?
   - Compare learned-model rollouts against true simulator rollouts.
   - Inspect control force magnitude, sign, smoothness, and saturation.
   - Check whether policy improvement in predicted rollouts transfers to the true environment.
   - Test multiple random seeds before trusting a single run.

## Quarter-Car Debugging Workflow

Use this order when diagnosing poor suspension performance:

1. Reproduce the issue with the smallest relevant script and a fixed seed.
2. Generate or inspect validation plots comparing active and passive rollouts on identical road input.
3. Check the cost breakdown: comfort, road holding, safety/barrier, and total cost.
4. Inspect control force over time and compute RMS/peak/saturation rate.
5. Check whether active force sign is physically sensible by trying simple sanity policies if needed.
6. Inspect GP one-step prediction error on held-out data.
7. Inspect multi-step model rollout error against the true simulator.
8. Only then change model learning, cost weights, policy structure, or optimization settings.

Prefer small diagnostic additions over large rewrites. Keep new experiments clearly named with `-run_name`, and avoid overwriting prior results unless the user explicitly wants that.

## Prior Dynamics and Residual Learning Guidance

For the quarter-car task, a useful prior is often available from linear suspension dynamics. The preferred pattern is residual learning:

```text
observed velocity increment = physics-prior velocity increment + GP residual
```

That means:

- During training, the GP target should be the residual between observed transition and prior-predicted transition.
- During prediction/rollout, the next-state update should add the physics prior and the GP residual.
- Deterministic integration relationships should remain deterministic where possible.
- The GP should learn what the approximate physical model cannot explain, not relearn all known mechanics from scratch.

Before adding a more complex kernel, first verify:

- The prior dynamics equation uses the same sign convention as the simulator.
- Physical parameters match the environment.
- Road displacement/velocity are aligned with state/input samples.
- The residual magnitude is smaller and easier to model than the raw velocity increment.

Relevant existing reference:

- `MODEL_PRIOR_GUIDE.md` explains how to introduce model prior knowledge and why residual learning is recommended for quarter-car suspension.

## Important Files and Entry Points

Core MC-PILCO:

- `policy_learning/MC_PILCO.py`: main training loop, policy optimization, logging, and data collection.
- `policy_learning/MC_PILCO_gym.py`: Gym/GOPS integration, road seed handling, validation rollout support, and road collection.
- `policy_learning/Policy.py`: policy classes and control parameterization.
- `policy_learning/Cost_function.py`: cost functions, including suspension evaluation cost.
- `model_learning/Model_learning.py`: GP model-learning classes, including quarter-car reconstruction and physics-residual variants.
- `simulation_class/gym_model.py`: Gym rollout wrapper and road-signal collection.
- `simulation_class/model.py`: ODE-based rollout wrapper.
- `simulation_class/ode_systems.py`: ODE system definitions.

Quarter-car scripts:

- `test_mcpilco_quarter_car.py`: ODE-style quarter-car test.
- `test_mcpilco_quarter_car_gym.py`: Gym/GOPS quarter-car training.
- `test_mcpilco_quarter_car_gym_reconstruct.py`: quarter-car state reconstruction variant.
- `test_mcpilco_quarter_car_gym_residual.py`: physics-informed residual model-learning variant.
- `log_plot_quarter_car.py`: validation plotting and active/passive metric comparison.
- `tune_cost_parameters.py`: cost-parameter sweep.
- `tune_hyperparameters.py`: training/model/policy hyperparameter sweep.
- `quarter_car_gui_launcher.py` and `web_dashboard_server.py`: user-facing launch/dashboard utilities.

Results:

- Training results are normally saved under `results_tmp/`.
- Quarter-car Gym experiments commonly use `results_tmp/quarter_car_gym/seed_<seed>/<run_name>/`.

## Running Tests and Experiments

Create the conda environment:

```bash
conda env create --file environment.yaml
conda activate mc-pilco
```

All test scripts are in the repository root and should be run from there.

Standard MC-PILCO tests:

```bash
python test_mcpilco_cartpole.py
python test_mcpilco_cartpole_rbf_ker.py
python test_mcpilco_cartpole_multi_init.py
```

MC-PILCO-4PMS:

```bash
python test_mcpilco4pms_cartpole.py
```

Quarter-car experiments:

```bash
python test_mcpilco_quarter_car.py
python test_mcpilco_quarter_car_gym.py -seed 42 -run_name debug_baseline
python test_mcpilco_quarter_car_gym_residual.py -seed 42 -run_name debug_residual
python log_plot_quarter_car.py -result_root results_tmp/quarter_car_gym -seed 42 -run_name debug_residual
```

MuJoCo environments require MuJoCo installation:

```bash
python test_mcpilco_cartpole_mujoco.py
python test_mcpilco_ur5_mujoco.py
```

Apply learned policies:

```bash
python apply_mcpilco_policy.py
python apply_mcpilco_policy_on_model.py
python apply_mcpilco4pms_policy.py
python apply_mcpilco4pms_policy_on_model.py
```

Most tests accept `-seed`, for example:

```bash
python test_mcpilco_quarter_car_gym_residual.py -seed 42
```

Use `repeat_test.py` or the tuning scripts when statistical comparison across seeds/configurations is required.

## Evidence to Collect Before Major Changes

When investigating bad performance, collect or compute:

- Active vs passive trajectories on the same validation road.
- RMS and peak values for sprung acceleration, suspension travel, tire deflection, and control force.
- Cost component curves and cumulative cost.
- Policy input/output ranges and actuator saturation rate.
- One-step GP prediction error.
- Multi-step model rollout error.
- Training cost curve and standard deviation curve.
- Performance across at least a few seeds when possible.

If a change improves training cost but worsens true validation cost, suspect model exploitation or cost mismatch.

## Common Failure Modes

Watch especially for:

- Active force sign inverted.
- Road profile differs between training, validation, and passive replay.
- Passive baseline uses different parameters or reset conditions.
- Observation scaling differs from physical units.
- The policy sees scaled states but the cost assumes physical units, or vice versa.
- GP target shape/order does not match state update assumptions.
- Road input is collected but not included in GP inputs, or included with wrong alignment.
- Physics prior uses inconsistent units, signs, or sample time.
- Cost weights make the controller sacrifice road holding or safety to reduce acceleration.
- Control force is saturated, too weak, or effectively zero.
- Policy rollout improves on the learned model but fails in the simulator.
- A single random seed is mistaken for a reliable result.

## Code Architecture

### Core Components

`policy_learning/` - Policy optimization and cost functions

- `MC_PILCO.py`: main MC-PILCO class implementing the reinforcement learning loop.
  - `reinforce()`: main training loop with exploration and policy optimization phases.
  - Manages data collection, model learning, and policy updates.
- `MC_PILCO_gym.py`: MC-PILCO variant for Gym/GOPS-style environments.
- `MC_PILCO_mujoco_envs.py`: MC-PILCO variant for MuJoCo environments.
- `Policy.py`: policy classes such as `Random_exploration`, `RBF_Policy`, and `Linear_Policy`.
- `Cost_function.py`: cost function implementations for different tasks.

`model_learning/` - Gaussian Process model learning

- `Model_learning.py`: GP-based system dynamics learning.
  - Supports exact GP inference and approximations such as SOR and SOD.
  - Includes velocity/state-difference model-learning variants.
  - Includes quarter-car reconstruction and physics-residual model-learning variants.
  - Handles angle wrapping and state transformations for original benchmark tasks.

`simulation_class/` - System simulation

- `model.py`: ODE-based system simulation wrapper.
- `gym_model.py`: Gym/GOPS environment wrapper.
- `model_mujoco.py`: MuJoCo environment wrapper.
- `ode_systems.py`: ODE definitions for simulated systems.
- `road_profiles.py`: road-profile generation utilities when present/used by plotting or simulation.

`gpr_lib/` - Gaussian Process Regression library

- `GP_prior/`: GP prior implementations.
- `Likelihood/`: likelihood functions.
- `Utils/`: covariance functions and scaling utilities.

`envs/` - Environment definitions

- `cartpole_swingup.py`: cart-pole environment for Gym/MuJoCo.
- `ur5.py`: UR5 robot environment.
- `assets/`: MuJoCo XML model files.

## Key Workflow

1. Exploration phase: random policy collects initial data from the system.
2. Model learning: GPs learn system dynamics from collected data.
3. Policy optimization: Monte Carlo gradient estimation optimizes policy parameters.
4. Control phase: optimized policy is applied and new data is collected.
5. Iteration: steps 2-4 repeat for multiple trials.

For quarter-car active suspension, add an explicit validation habit after each trial/configuration:

1. Replay active policy and passive baseline on the same fixed road.
2. Compare comfort, road holding, safety, and control effort.
3. Check whether learned-model improvement transfers to true simulator improvement.

## Code Style

- Formatting: Black with 120 character line length.
- Import sorting: isort with Black profile.
- Linting: flake8 is warnings-only and should not fail pre-commit.
- Pre-commit hooks: run `pre-commit install` to enable automatic formatting.

Format code:

```bash
black --line-length=120 <file>
isort --profile black --line-length 120 <file>
```

## Working Practices

- Inspect existing results and scripts before modifying code.
- Preserve user experiments and logs; do not delete or overwrite result directories unless explicitly asked.
- Keep changes small and reversible.
- Prefer adding diagnostics, assertions, and plots before changing algorithms.
- Use fixed seeds and explicit `-run_name` values for reproducibility.
- If running a long experiment is necessary, explain what it is expected to prove.
- When giving a diagnosis, include:
  - what was checked,
  - what evidence was found,
  - the most likely cause,
  - confidence level,
  - the next recommended experiment or code change.

## Logging and Results

- Test scripts save results to `results_tmp/` by default.
- Use `log_plot_*.py` scripts to visualize results:
  - `log_plot_cartpole.py`: plot cart-pole results.
  - `log_plot_cartpole_mujoco.py`: plot MuJoCo cart-pole results.
  - `log_plot_quarter_car.py`: plot quarter-car validation results and active/passive comparisons.
  - `log_plot_ur5.py`: plot UR5 results.

## License

AGPL-3.0-or-later. Preserve existing MERL copyright/license headers when editing source files.
