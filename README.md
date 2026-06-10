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

Below steps denotes the 1st appraoch we have used using MARL,

1. Train n number of RL agents (Candidate Agents) Agent_c1 , Agent_c2 , ..Agent_cn on a given environment env with their own copy of the environment and set their goal to improve the code reduction
2. Train m number of RL agents (Candidate Agents) Agent_e1 , Agent_e2 , ..Agent_en on a given environment env (same env as the first step) with their own copy of the environment and set their goal to improve the execution time.

![](/assets/images/approach1.1.png)


3. Define the primary objective and the secondary objective, these could be either the code reduction or the execution time reduction. The primary objective is the goal that the Main Agent is trained toward.
4. We use the trained Candidate Agents Agent_c1 , Agent_c2 , ..Agent_cn and Agent_e1 , Agent_e2 , ..Agent_en to train a Main RL Agent Agent_M. The goal of the Main Agent is to improve the primary objective and the reward calculation is set accordingly. We integrate the Candidate Agents by modifying the action space At of Agent_M . The action space in each time step t of Agent_M is set to the list of each prediction taken by each candidate agent for the observations and reward of the environment as shown in below.

![](/assets/images/approach1.2.png)


5. The trained model provided by the main agent Agent_M is used to evaluate the goodness of the approach by using standard benchmarks.

Reward function - R = α ∗ R_BinSize + β ∗ R_Throughput

α - Size coefficient
β - Througput coefficient

## Approach 2 - Using Multiple RL Agents with Split Action Space to Train Main RL Agent

