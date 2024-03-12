% This file runs the Markov process for many burn-in periods until the aggregate distribution across engines converges to the "invariant distribution",
% saving the subsequent observations in stdist_draws.mat.
%
% We run the DGP for an initial amount of time (the burn-in periods, which we discard) so that the average agent approaches equilibrium (time stationarity),
% where the distribution across engines (of any variable) approaches the "invariant distributions" (the limit as the number of time periods goes to infinity).
% (The time stationarity results from the ergodicity of the specific Markov process used here.)
% Then, by the ergodicity of the specific Markov process used, we can just aggregate over the remaining periods and engines to get the invariant distribution.
%
% AUTHOR
% William Liu (liuw@mit.edu) 2024

%% Preliminaries
clear
parpool('local', str2double(getenv('SLURM_CPUS_PER_TASK')))  % For HPC cluster
rng(2024);  % Set seed

%* Set code parameter values
% We want to minimize the number of engines in our sample because we will "waste" burn-in periods on fewer engines;
% we can have a high number of total time periods to compensate.
% Reflecting this, each Monte Carlo "sample" has one engine; we only have different Monte Carlo samples (and more than one engine) to
% make full use of parallel computing resources.
% The "real" sample is the aggregate of all these individual "samples."
periods = 1000000;  % Kept time periods
burn_in = 1000000;  % Burn-in periods
draws = 64;  % Number of "samples," engines, and parallel workers/processes

%* Create params struct
size_w = 4;  % Number of iid state variables that affect transitions
params.maintenance_factor = -0.5;  % Per-period maintenance cost coefficient
params.replacement_cost = -1;  % Replacement cost
params.beta = 0.9;  % Discount factor
params.coeff_w = 0.1 * (1:size_w).^-2;  % Coefficients c_k

%% Generate simulated data
load("cond_prob_repl_fn.mat")
data = nan(periods, 2+size_w, draws);

parfor j = 1:draws
    [~, datatmp] = gen_data_fn(cond_prob_repl_fn, params, 1, periods-1, burn_in);  % "periods-1" + 1 number of periods are generated
    data(:, :, j) = datatmp;
end

stdist_draws = reshape(data(:, 2, :), 1, []);  % Aggregate mileage points over time periods and "samples" (i.e., engines)

save("stdist_draws.mat", "stdist_draws", "-v7.3");
% save("stdist_draws_v6.mat", "stdist_draws", "-v6")  % For use in R for convergence diagnostics