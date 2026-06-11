# Multi Agent Dynamic Action Space Solution for Phase Ordering

This repository includes the step by step Scripts used to implement a Multi-Agent Dynamic Action solution for Compiler Optimization - Phase Ordering problem using CompilerGym, StableBaselines3, and Optuna.


## Phase Ordering Problem

In the compiler optimization domain, Phase Ordering is the problem of selecting a sequence of optimization actions that could be applied to the intermediate code to optimize the program. Optimizing could refer to either reducing the execution time, reducing the code size, or reducing the power consumption. Finding the optimal answer is complex because one optimization action could be used multiple times to reach the optimal solution, and the number of actions in a sequence could vary depending on the intermediate code. Loop Unrolling, Dead Code Elimination, Function Inlining, Loop Sink, and Dead Argument Elimination are examples of optimization actions

In the CompilerGym framework for the LLVM environment, 124 possible optimization actions could be used to optimize a program. The best possible action sequence that provides the best optimizations could be in the type of the following (Actions are named A, B, C, D, E),

1. E - A - C - D - B , Sequence with non-repeated actions
2. E - E - A - A - B - D - D, Sequence with repeated actions
3. A - B, Short Sequence, which has no need to use other actions, ie. Other
actions have no effect on the IR
4. B - C - D - A - A - B - A - B - A - B - A - B - C , Long Sequence which
needs multiple occurrences of actions to provide a valid result

### Mapping the phase ordering problem into RL 

1. Observation Space ( S ) - Characteristics of the IR code. Example (Number of Load instructions, Number of MUL instructions, Number of OR instructions, Number of Unary operations, Number of calls that return an int)

2. Reward Space ( R ) - Reward could be either the Code size reduction ratio, compilation time reduction ratio or a function of both

3. Action Space ( A ) - Set of Optimization methods that build up the sequence

## Approach 1 - Using Multiple RL Agents with Different Rewards to Improve the Execution Time along with Code Size Reduction

The aim of this Approach is to Maximize a main reward along with a secondary reward. (Reducing the code size while improving the compilation time) Below steps denotes the 1st appraoch we have used using MARL,

1. Train n number of RL agents (Candidate Agents) Agent_c1 , Agent_c2 , ..Agent_cn on a given environment env with their own copy of the environment and set their goal to improve the code reduction
2. Train m number of RL agents (Candidate Agents) Agent_e1 , Agent_e2 , ..Agent_en on a given environment env (same env as the first step) with their own copy of the environment and set their goal to improve the execution time.

![](/assets/images/approach1.1.png)


3. Define the primary objective and the secondary objective, these could be either the code reduction or the execution time reduction. The primary objective is the goal that the Main Agent is trained toward.
4. We use the trained Candidate Agents Agent_c1 , Agent_c2 , ..Agent_cn and Agent_e1 , Agent_e2 , ..Agent_en to train a Main RL Agent Agent_M. The goal of the Main Agent is to improve the primary objective and the reward calculation is set accordingly. We integrate the Candidate Agents by modifying the action space At of Agent_M . The action space in each time step t of Agent_M is set to the list of each prediction taken by each candidate agent for the observations and reward of the environment as shown in below.

![](/assets/images/approach1.2.png)


5. The trained model provided by the main agent Agent_M is used to evaluate the goodness of the approach by using standard benchmarks.

### Running Approach 1

The generic Approach 1 configuration is
`configs/approach1_bzip2.json`. It defines each candidate agent independently,
including its objective, sequence length, training budget, architecture, and
hyperparameters. The supplied configuration reproduces the thesis structure:

- Runtime candidate with a 40-action sequence.
- Runtime candidate with a 5-action sequence.
- Code-size candidate with a 40-action sequence.
- Code-size candidate with a 5-action sequence.

Run it with:

```bash
python3 Approach1/Run_Approach1.py \
  --config configs/approach1_bzip2.json
```

At each main-agent timestep, the main action selects one candidate agent. That
candidate observes the current main environment state and predicts an LLVM
optimization pass. The predicted LLVM pass, rather than the candidate index,
is then applied to the main environment.

Results are written to:

```text
results/approach1_bzip2/seed_0/
```

The directory contains:

- `children/*.zip`: every trained candidate model.
- `approach1_bzip2_main_<timesteps>_steps.zip`: final main selector model.
- `evaluation_episodes.csv`: per-episode main-model evaluation data.
- `evaluation_summary.csv`: timestep summaries compatible with
  `experiment.plot_results`.
- `metadata.json`: complete Approach 1 configuration and package versions.
- `run_summary.json`: child and main model paths and actual timestep counts.

The `runtime_reward` configuration supports:

- `"negative"`: reward is negative absolute runtime, matching the original
  Approach 1 script.
- `"delta"`: reward is the runtime improvement from the previous state.

Reload the saved main and child models for a later evaluation:

```bash
python3 Approach1/Evaluate_Approach1.py \
  --run-dir results/approach1_bzip2/seed_0 \
  --benchmark cbench-v1/qsort \
  --episodes 100
```

The evaluator reconstructs the candidate-selector environment, loads every
child model listed in `run_summary.json`, and then loads the main model. Its
CSV output uses the same format as normal experiments and can be passed
directly to `experiment.plot_results`.


## Approach 2 - Using Multiple RL Agents with Split Action Space to Train Main RL Agent

he aim of this Approach is tachieve a higher reward with fewer timesteps Below steps denotes the 1st appraoch we have used using MARL,

1. Define the number of Candidate agents n and divide the full action space (A) into n number of sub sets. Currently, splitting of the action space is done randomly. n ∈ [1, 5].
Complete Action Space : A ∈ {A1 , A2 , A3 , A5 ..An }
Subsets: A1 ∈ {A4 , A6 ..}
A2 ∈ {A5 , A7 , A100 ..}
An ∈ {A1 , A9 , A120 ..An }

2. Train n number of RL agents (Candidate Agents) Agentc1 , Agentc2 , ..Agentcn on a given environment env with their own copy of the environment and
set their goal to improve the code reduction. The goal is integrated into the environment by setting the reward of the environment related to the
code size reduction. When training each agent, the action space is limited to each action space subset which was generated in step 1.

![](/assets/images/approach2.1.png)

3. We use the trained Candidate Agents Agentc1 , Agentc2 , ..Agentcn to train the Main Agent Agent_M in a copy of the same environment. The goal of the Main Agent is to optimize the code reduction,and the reward calculation is set accordingly. We integrate the Candidate Agents by modifying the action space A_t of Agent_M . The action space in each time step t of Agent_M is set to the list of each prediction taken by each candidate agent for the observations and reward of the environment as shown below.

![](/assets/images/approach2.2.png)

4. The trained model provided by the main agent Agent_M is used to evaluate the goodness of the approach by using standard benchmarks.

### Running Approach 2

The generic configuration is `configs/approach2_bzip2.json`. It reproduces the
thesis structure with five candidate agents and one main selector agent:

1. Discover the complete LLVM action space from CompilerGym.
2. Shuffle all action IDs using `subset_seed`.
3. Divide every action into balanced, disjoint random subsets.
4. Train one candidate model using only the actions in its subset.
5. Train the main model with one action per candidate.
6. At each main timestep, the selected candidate predicts a local subset
   action, which is mapped back to the corresponding LLVM action and executed.

Run it with:

```bash
python3 Approach2/Run_Approach2.py \
  --config configs/approach2_bzip2.json
```

Use a different `subset_seed` to generate another random partition, such as the
two `Mc1` and `Mc2` rounds described in the thesis. Use a different experiment
`name` to preserve both result sets.

Results are written to:

```text
results/approach2_bzip2/seed_0/
```

The directory contains:

- `action_subsets.json`: exact random subsets and subset seed.
- `children/*.zip`: one trained candidate model per subset.
- `approach2_bzip2_main_<timesteps>_steps.zip`: final main selector model.
- `evaluation_episodes.csv` and `evaluation_summary.csv`.
- `metadata.json` and `run_summary.json`.

Reload the children, subsets, and main model for later evaluation:

```bash
python3 Approach2/Evaluate_Approach2.py \
  --run-dir results/approach2_bzip2/seed_0 \
  --benchmark cbench-v1/qsort \
  --episodes 100
```

Approach 2 summaries use the standard result schema and can be compared against
the baseline and Approach 1 using `experiment.plot_results`.

## Implementation and Evaluation

Framework used : CompilerGym , Stable Baselines3
HyperParameter Tuning : Optuna
Used RL Algorithms : PPO.

I have implemented a generic python script that can be modified for different parameters in ScriptsToGenerateBasicRLModelsWithDifferentConfig/ExperimentalSetup.py
### Base case

Environment : LLVM

Action Space : Entire Action Space

Reward : InstCount

Observation Space : Autophase

RL Algorithm : PPO

Trained timesteps : 20000 (at most 500 full-length 40-action episodes)

Action Count for a Sequence : 40

Fully Connected Layers : 4, with 64 nodes for each layer

Hyperparameters : Default without tuning

Training Data Set : cBench/Bzip2

**Evaluation**

Training and evaluation use separate environment
instances, and evaluation data is written under `results/<name>/seed_<seed>/`
with:

- Per-episode measurements in `evaluation_episodes.csv`.
- Mean, median, standard deviation, and 95% confidence intervals in
  `evaluation_summary.csv`.
- The complete configuration, package versions, platform, and LLVM version in
  `metadata.json`.
- Requested and actual SB3 timestep counts in `run_summary.json`.

Each evaluation summary also records the actual number of training episodes
completed at that checkpoint.

PPO and A2C complete whole rollout buffers, so their actual timestep count may
slightly exceed the configured target. CSV checkpoints and model filenames use
SB3's actual counter. Re-running the same experiment name and seed replaces its
generated evaluation CSV files instead of mixing runs.

Run the example configuration:

```bash
python3 ScriptsToGenerateBasicRLModelsWithDifferentConfig/ExperimentalSetup.py \
  --config configs/ppo_bzip2.json
```

The final model is always saved as:

```text
results/<name>/seed_<seed>/<name>_<actual-timesteps>_steps.zip
```

Set `"save_checkpoints": true` in the JSON configuration, or add
`--save-checkpoints` to the command, to save a model ZIP after every evaluation
checkpoint.

Load a saved model and evaluate it later:

```bash
python3 -m experiment.evaluate_saved \
  --model results/ppo_bzip2/seed_0/ppo_bzip2_20480_steps.zip \
  --benchmark cbench-v1/qsort \
  --benchmark cbench-v1/susan \
  --episodes 100
```

When `metadata.json` is beside the model, the evaluator automatically restores
the algorithm, observation space, action limit, default benchmarks, seed, and
evaluation settings. Use `--config configs/ppo_bzip2.json` when evaluating a
model stored elsewhere.

By default, loaded-model results are written to a timestamped directory:

```text
<model-directory>/manual_evaluations/<timestamp>/
```

Use `--output-dir` to choose a fixed location and `--stochastic` to sample
actions instead of using deterministic predictions.

The example uses Bzip2 only for training and uses separate cBench programs for
validation. For final reporting, create a second configuration containing
held-out test benchmarks and do not use those results to select architecture,
hyperparameters, or sequence length.

Run independent replications by changing `seed` in the configuration. At least
five seeds should be used for reported results. Confidence intervals from one
seed describe evaluation-episode variation; conclusions about training
stability must aggregate independently trained seeds.

Aggregate the final checkpoint from all `seed_*` directories:

```bash
python3 -m experiment.aggregate results/ppo_bzip2 \
  --output results/ppo_bzip2/aggregate_seeds.csv
```

Plot code size and runtime against training timesteps:

```bash
python3 -m experiment.plot_results \
  --result results/ppo_default \
  --label "PPO Default" \
  --result results/ppo_tuned \
  --label "PPO Tuned" \
  --measurement codesize \
  --measurement runtime \
  --output-dir charts/ppo_comparison
```

Each `--result` may point to an experiment folder containing `seed_*`
directories, a single seed folder, or an `evaluation_summary.csv` file. When
multiple seeds are found, values are averaged at each timestep and the shaded
region shows the cross-seed 95% confidence interval.

The script generates one chart per benchmark and measurement, for example:

```text
charts/ppo_comparison/cbench-v1-qsort_codesize_vs_timesteps.png
charts/ppo_comparison/cbench-v1-qsort_runtime_vs_timesteps.png
```

Use only one measurement when needed:

```bash
python3 -m experiment.plot_results \
  --result results/ppo_bzip2/seed_0 \
  --measurement codesize \
  --benchmark cbench-v1/qsort \
  --output-dir charts/bzip2
```

Available options include `--no-ci`, `--format pdf`, `--format svg`,
`--title-prefix`, and repeated `--benchmark` filters.

Plot code size and runtime together on the same Y-axis:

```bash
python3 -m experiment.plot_results \
  --result results/ppo_bzip2 \
  --label "PPO" \
  --measurement combined \
  --runtime-scale 1000000 \
  --output-dir charts/combined
```

In this mode, instruction count is plotted without transformation and runtime
is plotted as `runtime x runtime-scale`. For example, `0.008` seconds becomes
`8000` when the scale is `1000000`. The chart legend and Y-axis label include
the multiplier so the scaled runtime is not confused with raw seconds.

Tune PPO against validation benchmarks:

```bash
python3 HyperParameterTuning/HyperParameterTuning.py \
  --config configs/ppo_bzip2.json \
  --trial-seed 0 --trial-seed 1 --trial-seed 2
```

The Optuna study is persisted in SQLite and tunes learning rate, rollout steps,
discount factor, GAE lambda, and clipping range. Copy the selected
`hyperparameters` object into a frozen final experiment configuration before
running held-out tests.

hyperparameters could be added to experiments as following.

```bash
"hyperparameters": {
    "learning_rate": 0.000215,
    "n_steps": 768,
    "gamma": 0.904995,
    "gae_lambda": 0.88688,
    "clip_range": 0.2
  }
  ```

Evaluate a seeded random-search baseline with the same sequence length:

```bash
python3 -m experiment.random_baseline \
  --benchmark cbench-v1/qsort \
  --benchmark cbench-v1/susan \
  --episodes 100 --max-episode-steps 40 --seed 0
```

Runtime results remain sensitive to system noise. Final runtime experiments
should pin CPU affinity and frequency, include warm-up runs, repeat each
measurement, and report medians and confidence intervals alongside the exact
hardware and software metadata.
