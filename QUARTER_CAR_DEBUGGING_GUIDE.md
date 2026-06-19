# Quarter-Car MC-PILCO Debugging Guide

This guide is a practical runbook for agents working on the quarter-car active suspension experiments. Use it when the learned active controller performs poorly or appears worse than the passive baseline.

The guiding rule is simple: do not change the algorithm until the failure is measured under a fair active/passive comparison.

## 1. Prepare the environment

Run all commands from the repository root.

    conda activate mc-pilco

If the environment does not exist yet:

    conda env create --file environment.yaml
    conda activate mc-pilco

For Gym/GOPS quarter-car experiments, confirm that the GOPS environment path used by the scripts is available on the machine. Some diagnostics can still read logs without replaying the environment, but passive replay and force-sign checks need the Gym environment.

Prefer GPU training when CUDA is available. The quarter-car residual training script already defaults to CUDA if PyTorch can see a GPU, but agents should still make the device explicit in important runs so that the result metadata is unambiguous:

    python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu')"

If CUDA is available, pass:

    -device cuda

If CUDA is unavailable or debugging a CPU-only failure, pass:

    -device cpu

## 2. Start with a small reproducible training run

Prefer the residual variant first because the suspension task has useful prior dynamics knowledge.

    python test_mcpilco_quarter_car_gym_residual.py -seed 1 -run_name debug_residual_s1 -use_suspension_cost

For GPU training, make the device explicit:

    python test_mcpilco_quarter_car_gym_residual.py -seed 1 -run_name debug_residual_s1 -use_suspension_cost -device cuda

For a small multi-seed check, use the same -run_name across seeds:

    python test_mcpilco_quarter_car_gym_residual.py -seed 1 -run_name debug_residual_v1 -use_suspension_cost
    python test_mcpilco_quarter_car_gym_residual.py -seed 2 -run_name debug_residual_v1 -use_suspension_cost
    python test_mcpilco_quarter_car_gym_residual.py -seed 3 -run_name debug_residual_v1 -use_suspension_cost

Results are saved under:

    results_tmp/quarter_car_gym/seed_<seed>/<run_name>/

Do not overwrite an old run unless the user explicitly asks. Use a new -run_name for every hypothesis.

Before launching a comparison batch, decide which parameters are fixed and which one is under test. Keep all non-tested parameters identical across runs.

Always record or verify these fields for each run:

- seed, road_seed, validation_road_seed, validation_G0;
- device, dtype, num_threads;
- num_trials, T_exploration, T_control, T_sampling;
- model_epochs, GP approximation settings, use_road_gp_input;
- policy num_basis, num_particles, lr, opt_steps, p_dropout;
- act_scaling, act_max, obs_scaling;
- physical parameters Ms, Mu, Ks, Cs, Kt;
- cost mode and cost parameters.

For suspension evaluation cost, record the exact cost proportions and scales:

- w_acc, w_tire, w_barrier;
- l_acc, l_tire, d_barrier, beta_barrier;
- if using the older saturated-distance cost, record cost_l0, cost_l_xs, cost_l1, cost_l2, cost_l3.

The scripts already save much of this in config_log.pkl and experiment_info.json. Treat those files as the source of truth for later parameter optimization.

## 3. Generate plots and the diagnostic table

First generate the validation plots:

    python log_plot_quarter_car.py -result_root results_tmp/quarter_car_gym -seed 1 -run_name debug_residual_s1

Then generate the active/passive diagnostic table:

    python diagnose_quarter_car_run.py -result_root results_tmp/quarter_car_gym -seed 1 -run_name debug_residual_s1

The diagnostic script writes:

    quarter_car_diagnostics.csv
    quarter_car_diagnostics.pkl
    quarter_car_diagnostics_summary.txt

Use quarter_car_diagnostics_summary.txt for a quick read. Use the CSV when comparing multiple validation rollouts or multiple seeds.

For parameter optimization, keep the diagnostic CSV/PKL together with the original config files:

    config_log.pkl
    experiment_info.json
    quarter_car_diagnostics.csv
    quarter_car_diagnostics.pkl
    quarter_car_diagnostics_summary.txt

Together, these files provide one row of evidence: what was trained, with which cost proportions and hyperparameters, and what happened on the fixed validation road.

If GOPS/MATLAB replay is unavailable but you still want log-only active metrics and residual-prior audit:

    python diagnose_quarter_car_run.py -result_root results_tmp/quarter_car_gym -seed 1 -run_name debug_residual_s1 -skip_env_replay

This mode cannot produce a true passive replay, so do not use it as final evidence that active beats passive.

## 4. Fairness audit before interpretation

Before judging performance, verify the comparison is fair:

- active and passive use the same fixed validation road seed;
- active and passive start from the same initial state;
- active and passive use the same horizon and sampling time;
- passive replay uses zero active force;
- physical parameters and observation/action scaling match the training config;
- the reported metrics are computed in physical units.

In quarter_car_diagnostics_summary.txt, check:

    source
    validation_config
    same_initial_state_for_passive
    same_reset_kwargs_for_passive
    passive_zero_force_replay
    env_replay_error

If passive replay is unavailable or env_replay_error is not empty, fix the replay/environment issue before drawing conclusions.

Also check that compared runs use the same cost definition. A run trained with Expected_suspension_evaluation_cost should not be compared directly against a run trained with the older saturated-distance cost unless the report explicitly says the training objectives differ.

## 5. Read the result metrics

Focus on these fields in quarter_car_diagnostics.csv:

    rms_sprung_accel_active
    rms_sprung_accel_passive
    rms_sprung_accel_improvement_pct
    rms_suspension_travel_active
    rms_suspension_travel_passive
    rms_suspension_travel_improvement_pct
    rms_tire_deflection_active
    rms_tire_deflection_passive
    rms_tire_deflection_improvement_pct
    mean_evaluation_cost_active
    mean_evaluation_cost_passive
    mean_evaluation_cost_improvement_pct
    rms_control_force_active
    peak_control_force_active
    control_saturation_rate

Positive improvement_pct means active is better than passive for that metric. Negative means active is worse.

Interpretation guide:

- Comfort improved but tire/travel worsened: cost weights likely over-favor sprung acceleration.
- Total cost improved but physical safety metrics worsened: evaluation cost may not reflect the real objective.
- Training cost improved but validation cost worsened: suspect model exploitation.
- Control force near zero and active does not improve: policy is too weak, action scaling is wrong, or effort penalties are too strong.
- Control force saturates often and active worsens: suspect force sign, model error, or overly aggressive policy optimization.

When comparing cost-tuning runs, do not only compare mean_evaluation_cost. Also compare the component means:

    mean_cost_comfort_active
    mean_cost_road_holding_active
    mean_cost_safety_active
    mean_cost_comfort_passive
    mean_cost_road_holding_passive
    mean_cost_safety_passive

These component values are the basis for deciding whether the cost proportions should be changed. For example, if comfort improves but road-holding cost rises sharply, increase the relative tire/road-holding weight or reduce the aggressiveness of acceleration-only optimization.

## 6. Check control-force sign

Run a short constant-force sanity check when the Gym environment can be loaded:

    python diagnose_quarter_car_run.py -result_root results_tmp/quarter_car_gym -seed 1 -run_name debug_residual_s1 -force_check

This writes:

    quarter_car_diagnostics_force_check.csv

Compare zero, positive, and negative force rows. The immediate sprung acceleration and suspension travel response should be physically sensible. If positive and negative force effects are reversed relative to the simulator convention, fix force sign or action scaling before changing cost or GP settings.

## 7. Audit the residual physics prior

The diagnostic script reports:

    raw_delta_rms
    physics_delta_rms
    residual_delta_rms
    residual_to_raw_ratio
    residual_smaller_than_raw_all_dims
    uses_road_gp_input
    road_exogenous_alignment

The residual prior is helping only if residual deltas are generally smaller than raw observation deltas. If residual_to_raw_ratio is near or above 1 in important dimensions, the prior is not simplifying the GP target.

Pay special attention to the current residual implementation:

- the prior assumes z_r = 0 and z_r_dot = 0;
- the prior sets tire force F_tire = 0;
- road input may be appended to GP inputs, but the physics prior itself does not use road displacement or road velocity.

If this audit fails, the first model-side hypothesis should be:

    the GP is being forced to learn too much road/tire dynamics as residual

The next code change should be small: include road/tire force in the physics prior and then rerun the same residual audit before running many training seeds.

## 8. Decide what to change

Use this decision order.

1. Fairness failed:
   - Fix validation reset, road seed, passive replay, or scaling.
   - Do not tune cost or model yet.

2. Force sign failed:
   - Fix action scaling/sign convention.
   - Rerun a tiny sanity experiment before full training.

3. Cost alignment failed:
   - Keep the model unchanged.
   - Adjust Expected_suspension_evaluation_cost weights or lengthscales.
   - Change one weight group at a time.
   - Keep a small table of tested cost proportions and validation metrics.

4. Residual prior audit failed:
   - Keep policy optimization unchanged.
   - Improve the physics prior target, especially road/tire dynamics.
   - Verify residual RMS decreases before training again.

5. Model transfer failed:
   - Compare learned-model rollout against true Gym rollout.
   - Reduce policy optimization aggressiveness or improve GP inputs/targets.

6. Single seed looks good or bad:
   - Do not conclude yet.
   - Run at least seeds 1, 2, and 3 with the same config.

7. GPU/CPU mismatch appears:
   - Confirm that device is recorded in config_log.pkl.
   - Reproduce one short run on the intended device before treating timing or numerical differences as algorithmic evidence.

## 9. Make one hypothesis-driven code change

When editing code, write down the hypothesis first. Good examples:

    Hypothesis: active is worse because the residual prior ignores road/tire dynamics.
    Change: add tire force using road displacement/velocity to the residual physics delta.
    Evidence expected: residual_to_raw_ratio decreases before policy training.

    Hypothesis: active reduces acceleration by sacrificing suspension travel.
    Change: increase safety/barrier or tire-deflection weight in the evaluation cost.
    Evidence expected: travel/tire RMS improve without large comfort regression.

Avoid combining model, cost, and policy changes in one run. If the result improves, you need to know why.

For hyperparameter or cost optimization, every candidate run should map to a clear parameter vector:

    run_name
    seed
    device
    use_suspension_cost
    w_acc, w_tire, w_barrier
    l_acc, l_tire, d_barrier, beta_barrier
    model_epochs
    num_particles
    lr
    opt_steps
    p_dropout
    num_basis
    use_road_gp_input
    validation_road_seed
    mean_evaluation_cost_improvement_pct
    rms_sprung_accel_improvement_pct
    rms_suspension_travel_improvement_pct
    rms_tire_deflection_improvement_pct
    control_saturation_rate

Use config_log.pkl and quarter_car_diagnostics.pkl/CSV to build this table. This table is the foundation for later automated parameter optimization.

## 10. Report results

When reporting back to the user, separate evidence from interpretation:

- What command was run.
- Which run directory was analyzed.
- Whether passive replay was fair and available.
- Device used for training, especially whether GPU/CUDA was used.
- Cost mode and exact cost proportions.
- Main model/policy hyperparameters.
- Active/passive percentage changes for comfort, suspension travel, tire deflection, and total cost.
- Control RMS/peak/saturation.
- Residual-prior audit result.
- Most likely cause.
- Confidence level.
- Next recommended experiment or code change.

Example report skeleton:

    Run: results_tmp/quarter_car_gym/seed_1/debug_residual_s1
    Device: cuda
    Cost: Expected_suspension_evaluation_cost, w_acc=0.4, w_tire=0.4, w_barrier=0.2
    Policy/model: lr=0.005, opt_steps=50, num_particles=400, model_epochs=2, road GP input enabled.
    Fairness: fixed validation road, same reset kwargs, passive zero-force replay available.
    Observed: active comfort +12%, tire deflection -8%, suspension travel -25%, total cost -4%.
    Control: RMS 640 N, peak 1000 N, saturation 31%.
    Residual prior: residual_to_raw_ratio > 1 in deflection velocity.
    Diagnosis: active likely exploits an inaccurate residual model and uses aggressive saturated force.
    Next: test force sign; then add road/tire force to residual prior before more cost tuning.

## 11. Useful files

- test_mcpilco_quarter_car_gym_residual.py: physics-informed residual training entrypoint.
- test_mcpilco_quarter_car_gym.py: standard Gym quarter-car training entrypoint.
- log_plot_quarter_car.py: validation plots and active/passive visual comparison.
- diagnose_quarter_car_run.py: active/passive metrics, control diagnostics, residual-prior audit.
- policy_learning/Cost_function.py: suspension evaluation cost.
- model_learning/Model_learning.py: GP model-learning and residual prior implementation.
- policy_learning/MC_PILCO_gym.py: fixed validation-road rollout and logging.
- MODEL_PRIOR_GUIDE.md: background on adding physics prior knowledge.
