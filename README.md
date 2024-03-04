This repository contains MATLAB code to replicate an updated version of Example 3 (Dynamic Discrete Choice) in the paper "Locally Robust Semiparametric Estimation" (Chernozhukov et al., 2022; subsection 3.2).

# DEPENDENCIES (MATLAB)
1. Statistics and Machine Learning Toolbox
2. Optimization Toolbox
3. Symbolic Math Toolbox
4. "All Permutations of integers with sum criteria" (Bruno Luong, 2024)

# INSTRUCTIONS FOR REPLICATING THE TABLES
1. Run solve_Markov_DP.m, which will solve the Markov decision problem.
    * This will use value iteration to solve an  "integrated Bellman equation" and will then plug the resulting "integrated value function" into a logit formula to
      get the conditional choice probablity function (for replacement), which is saved in cond_prob_repl_fn.m.
    * Example batch script for running on a computing cluster: LRSE_DP.sbatch.

2. Run get_stdist.m, which will get the stationary distribution of the Markov process for the engine mileage.
    * This runs the Markov process for many burn-in periods, saving the subsequent observations in stdist_draws.mat.
    * Example batch script for running on a computing cluster: LRSE_stdist.sbatch.

3. Set the variable "sample_size" and then run gen_data.m, which will generate the simulation datasets.
   In the paper, the sample sizes reported are 100, 300, 1000, and 10000; the script must be run separately for each of these sample sizes.
    * Example batch script for running on a computing cluster: LRSE_data.sbatch.

4. Set the variables "sample_size" and "spec" and then run estimatenew.m in order to estimate parameters using the simulated data.
   You can set "sample_size" to be 100, 300, 1000, or 10000 and "spec" to be 1, 1.5, 2, 3, or 4; the script must be run separately for each possible pair of values.
    * Example batch script for running on a computing cluster: LRSE_est.sbatch.

5. Run maketables.m (in the "Results" directory) to format the results as tables.

# CORRECTIONS FOR THE PAPER
1. The constant in H() should be Euler's constant. (The 2s and the 7 are the wrong way around.)

2. The bottom of page 1513 should say that there are 20 choices for gamma_1_ll'.
   This is because l and l' must be distinct (which is implied but not stated explicitly).

3. The expression for phi_1 should contain X_t+1 and Y_2t+1 instead of X_t and Y_2t.

4. The expression for phi_3 should contain Y_1t instead of Y_2t.

5. alpha_2 has two components corresponding to the two components of D, respectively.
   The conditional expectation of it in the formula for alpha_1 should actually be the sum of the two components of that conditional expectation.

# CHANGES TO THE EXAMPLE 3 SETUP
1. For the Markov transitions, the shock is now drawn from a different mixture distribution: 1+z times a half-normal with variance 1, where z denotes SUM_k {c_k * X_t,k+1}.

2. The static component of the per-period utility function is now sqrt(1+a) rather than sqrt(a).

# REFERENCES
1. Bruno Luong (2024). All Permutations of integers with sum criteria
(https://www.mathworks.com/matlabcentral/fileexchange/17818-all-permutations-of-integers-with-sum-criteria), MATLAB Central File Exchange. Retrieved January 15, 2024.
