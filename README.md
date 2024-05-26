This repository contains code for a remake of Example 3 (Dynamic Discrete Choice) in the paper "Locally Robust Semiparametric Estimation" (Chernozhukov et al., 2022; subsection 3.2).

# INSTRUCTIONS
1. Make sure dependencies 1-3 are installed. If doing parallel computing, also install dependency 4. (Dependency 5 is included in this repository.)

2. Run solve_Markov_DP.m, which will solve the Markov decision problem.
   * This will use value iteration to solve an  "integrated Bellman equation" and will then plug the resulting "integrated value function" into a logit formula to
      get the conditional choice probablity function (for replacement), which is saved in cond_prob_repl_fn.m.

3. Run get_stdist.m, which will get the stationary distribution of the Markov process for the engine mileage.
   * This runs the Markov process for many burn-in periods, saving the subsequent observations in stdist_draws.mat.

4. Set the variable "sample_size" and then run gen_data.m, which will generate the simulation datasets.
   In the paper, the sample sizes reported are 100, 300, 1000, and 10000; the script must be run separately for each of these sample sizes.

5. Set the variables "sample_size" and "spec" and then run estimate.m, which will calculate estimates using the simulation datasets.
   You can set "sample_size" to be 100, 300, 1000, or 10000 and "spec" to be 1, 1.5, 2, 3, or 4; the script must be run separately for each possible pair of values.

6. Run maketables.m (in the "Results" directory) to format the results as tables.

### DEPENDENCIES (MATLAB)
1. Statistics and Machine Learning Toolbox
2. Optimization Toolbox
3. Symbolic Math Toolbox
4. Parallel Computing Toolbox
5. "All Permutations of integers with sum criteria" (Bruno Luong, 2024)

### COMPUTING CLUSTER SUPPORT
Steps 2-5 can be run on a HPC cluster and are written in a way that assumes doing so (although, they can be modified to run locally).
Using a HPC cluster is particularly useful for steps 4-5, which require running scripts multiple times;
LRSE_data.sbatch and LRSE_est.sbatch will use a Slurm job array to automatically create separate jobs for each choice.

Batch files for running on a HPC cluster are provided:
* LRSE_DP.sbatch
* LRSE_stdist.sbatch
* LRSE_data.sbatch
* LRSE_est.sbatch

Simply change the parameters in these batch files to fit your system and run them as with any Slurm batch job. The job array number corresponds to sample size (step 4) or sample size and specification pair (step 5).

# NOTES
### AUTHOR
William Liu (liuw@mit.com) 2024
* Original project by Ben Deaner (bdeaner@mit.edu or bendeaner@gmail.com) 2020

### CHANGES FROM ORIGINAL PROJECT
1. For the Markov transitions, the shock is now drawn from a different mixture distribution: $1+z$ times a half-normal with variance 1, where $z$ denotes $\sum\limits_{k=1}^{4} c_k X_{k+1,t}$. The discrete i.i.d. variables remain $\mathrm{Bernoulli}(0.5)$, but the distribution of the continuous i.i.d. variables is now $U(0, \sqrt{12})$ (variance-one uniform) rather than $\chi^2_1$.

2. The static component of the per-period utility function is now $\sqrt{1+X_{1t}}$ rather than $\sqrt{X_{1t}}$, where $X_{1t}$ denotes mileage.
   This also means that $D_2(X_t) = (0, \sqrt{1+X_{1t}})'$.

3. Fixed formula errors and numerical accuracy issues.

4. Number of Monte Carlo draws increased to 1000.

### ERRORS IN THE PAPER
1. The expression for $\phi_1$ should contain $X_{t+1}$ and $Y_{2,t+1}$ instead of $X_t$ and $Y_{2,t}$.

2. The expression for $\phi_3$ should contain $Y_{1,t}$ instead of $Y_{2,t}$.

3. The constant in $H$ should be Euler's constant. (The 2s and the 7 are the wrong way around.)

4. The bottom of page 1513 should say that there are 20 choices for $\gamma_{1,ll'}$.
   This is because $l$ and $l'$ must be distinct (which is implied but not stated explicitly).

5. $D_1(X_t)$ should equal $(1, 0)'$ rather than $(-1, 0)'$.

6. $\hat{a}$ should have $\tilde{\theta}_l$ as an argument, not $\hat{\theta}_l$ (which does not exist).

# REFERENCES
1. Chernozhukov, V., Escanciano, J.C., Ichimura, H., Newey, W.K. and Robins, J.M. (2022), Locally Robust Semiparametric Estimation. Econometrica, 90: 1501-1535. https://doi.org/10.3982/ECTA16294

2. Bruno Luong (2024). All Permutations of integers with sum criteria
(https://www.mathworks.com/matlabcentral/fileexchange/17818-all-permutations-of-integers-with-sum-criteria), MATLAB Central File Exchange. Retrieved January 15, 2024.