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