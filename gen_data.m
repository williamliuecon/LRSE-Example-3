% This file generates simulated datasets for use in the engine replacement Monte Carlo simulation.
%
% The simulated datasets are saved to the path "save_path". The saved output is a four-dimensional array.
% The rows of the final output correspond to different engines, the columns to different time periods, and the fourth dimension to different draws.
% The third dimension corresponds to variables of interest: choice data, mileage, the states that affect only transitions, the utility heterogeneity,
% and then the values of each of these variables for the next period. (This is convenient when doing the estimation.)
%
% AUTHOR
% William Liu (liuw@mit.edu) 2024

%% Setup
clear
rng(2024);  % Set seed

%* For HPC cluster
parpool("local", str2double(getenv("SLURM_CPUS_PER_TASK")))
task_id = str2double(getenv("SLURM_ARRAY_TASK_ID"));
values = [100, 300, 1000, 10000];
sample_size = values(task_id);

%* Code parameters
% sample_size = 10000;  % Number of individual engines in each sample (i.e., dataset)
periods = 10;  % T, the number of time periods of the design matrix (representing T+1 periods overall because of the lead variables)
draws = 1000;  % Number of Monte Carlo draws for the simulated data; each draw is a sample (i.e., dataset)

%* Create params struct
size_w = 4;  % Number of iid state variables that affect transitions
params.maintenance_factor = -0.5;  % Per-period maintenance cost coefficient
params.replacement_cost = -1;  % Replacement cost
params.beta = 0.9;  % Discount factor
params.coeff_w = 0.1 * (1:size_w).^-2;  % Coefficients c_k

%% Generate simulated data
% Rather than doing burn-in periods for all the engines in the final sample, it is much more efficient to first do burn-in periods for a few engines to
% get the time-stationary distribution and then do a new run, initialized at that distribution, with many engines.
load("cond_prob_repl_fn.mat")
load("stdist_draws.mat")
data = nan(sample_size, periods, 4+2*size_w, draws);
data_filename = strcat("data_t=", num2str(periods), "_n=", num2str(sample_size), ".mat");  % Do this here in case strange bug on cluster, whereby this line fails, occurs

parfor j = 1:draws
    init_mile = randsample(stdist_draws, sample_size, true)
    datatmp = gen_data_fn(cond_prob_repl_fn, params, sample_size, periods, 0, init_mile);
    data(:, :, :, j) = datatmp;
    fprintf("Draw number: %d\n", j)
end

save(data_filename, "data", "-v7.3");

%% Testing
%{
%* Generate 2D histogram of the mileage to check closeness to invariant distribution
load(data_filename)
data_cellvec = cell(draws, 1);
for j = 1:draws
    data_cellvec(j, 1) = {data(:, :, :, j)};
end

data_agg = cat(1, data_cellvec{:});
m_mat = data_agg(:, :, 2);
time_mat = repmat(1:periods, sample_size * draws, 1);
histogram2(m_mat, time_mat, "FaceColor", "flat")
xlabel("Mileage")
ylabel("Time")
%}