% This file carries out the estimation for the bus engine replacement Monte Carlo simulation.
%
% AUTHOR
% William Liu (liuw@mit.edu) 2024

% How "spec" determines the basis functions:
% * 1 – only linear terms
% * 1.5 – linear and squared terms
% * 2 – linear, squared terms, and interactions
% * >=2 – all powers and interactions up to that order (i.e., terms of a multinomial expansion)

%% Setup
clear
rng("default")  % Need to have a fixed seed because the lasso cross-validation introduces randomness

%* Code parameters
%// For running on a HPC cluster
% parpool("local", str2double(getenv("SLURM_CPUS_PER_TASK")))
% task_id = str2double(getenv("SLURM_ARRAY_TASK_ID"));
% value_set = { ...
%     [1, 100], [1.5, 100], [2, 100], [3, 100], [4, 100], ...
%     [1, 300], [1.5, 300], [2, 300], [3, 300], [4, 300],  ...
%     [1, 1000], [1.5, 1000], [2, 1000], [3, 1000], [4, 1000],  ...
%     [1, 10000], [1.5, 10000], [2, 10000], [3, 10000], [4, 10000] ...
% };
% value = value_set{task_id};
% spec = value(1);
% sample_size = value(2);

%// For running locally
parpool("local", 6)  % Set to however many cores you want to use for parallel computing
spec = 1;            % Set to 1, 1.5, 2, 3, or 4
sample_size = 100;   % Set to 100, 300, 1000, or 10000

%// For both
rule_of_thumb = 1;  % Set this to 1 to use rule-of-thumb penalty parameters instead of cross-fitting
periods = 10;
delta = 0.9;  % Discount factor
splitnum = 5;  % Number of folds to use for cross-fitting

%% Main code
%* Get data
data_filename = strcat("data_t=", num2str(periods), "_n=", num2str(sample_size), ".mat");
results_filename = strcat("est_t=", num2str(periods), "_n=", num2str(sample_size), "_", num2str(spec), ".mat");
load(data_filename);  % Load simulated data, generated using just_data.m

%* Calculate some objects
draws = size(data, 4);  % Number of Monte Carlo draws
periods = size(data, 2);  % Number of time periods in the design matrix
sample_size = size(data, 1);  % Number of individual engines in each Monte Carlo sample
size_x = size(data, 3)/2 - 1;  % Number of state variables (mileage and the iid variabless)

%* Truncate data (if necessary) so that resulting sample size is a multiple of splitnum
splitsize = floor(sample_size / splitnum);
if (sample_size / splitnum) - splitsize > 0
    data(splitsize*splitnum+1:end, :, :, :) = [];
end

%* Flatten the data
Data = permute(reshape(permute(data, [2, 1, 3, 4]), [size(data, 1)*size(data, 2), 1, size(data, 3), size(data, 4)]), [1, 3, 4, 2]);

%* Get indices of folds for sample splitting
I = logical(kron(eye(splitnum), ones(periods*splitsize,1)));  % The obs of each entity (engine) are indeed next to each other, so we are crossfitting across entities

%% Perform estimation for each Monte Carlo draw
est = nan(2, 2, draws);
sd = nan(2, 2, draws);
% The first dimension refers to the components of theta.
% The second dimension refers to the estimator (uncorrected and then corrected)

parfor i = 1:draws
    [est(:, :, i), sd(:, :, i)] = estimate_fn(Data(:, :, i), I, size_x, delta, spec, rule_of_thumb);             
    disp(i)
end

save(results_filename, "est", "sd", "-v7.3");
 
