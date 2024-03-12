% This function generates the data using the precalculated conditional choice probabilty (of replacement) function, cond_prob_repl_fn.
% The columns of the returned dataset are: an indicator for replacement, mileage, the iid variables, and then the next-period versions
% of all of these.
%
% AUTHOR
% William Liu (liuw@mit.edu) 2024

function [data, data_raw, durations] = gen_data_fn(cond_prob_repl_fn, params, sample_size, periods, burn_in, init_mile)
    %The arguments:
    % cond_prob_repl_fn – function returning the probability of replacement conditional on mileage and the iid variables
    % params – struct containing the regression single-index parameters "c_k" as well as other parameters
    % sample_size – number of individual engines in each sample (i.e., dataset)
    % periods – T, the number of time periods of the design matrix (representing T+1 periods overall because of the lead variables)
    % burn_in – burn-in periods, which are discarded
    % init_mile – vector of mileages to initialize the engines at for the Monte Carlo simulation

    size_w = size(params.coeff_w, 2);  % Number of iid state variables that affect transitions
    data_raw = nan(sample_size, periods+1, 2+size_w);
    total_obs = sample_size * (periods + burn_in + 1);
    
    %% Monte Carlo simulation of variables
    %* Simulate next-period iid state variables
    % The iid state variables are each Uniform or Bernoulli. They are arranged in alternating order:
    %     Uniform, Bernoulli, Uniform, Bernoulli, Uniform, ...
    w_unif = sqrt(12) * rand(total_obs, ceil(size_w/2));
    w_disc = binornd(1, 0.5, total_obs, floor(size_w/2));
    if mod(size_w, 2) ~= 0  % Interleave columns and then tranpose
        w_draws = cat(2, reshape([w_unif(:, 1:end-1); w_disc], size(w_unif, 1), []), w_unif(:, end));
    else
        w_draws = reshape([w_unif(:, 1:end); w_disc], size(w_unif, 1), []);
    end
    z_draws = w_draws * params.coeff_w';
    
    %* Simulate persistent state variable (mileage) and other data (choice, etc.)
    durations = [];
    for i = 1:sample_size
        if nargin == 6
            m = init_mile(i);
        else
            m = 0;
        end
        last_replacement = NaN;
        for t = 1:periods+burn_in+1 % Add burn-in periods
            index = (i-1)*(periods+burn_in+1)+t;
            
            %* Generate choice
            replace = binornd(1, cond_prob_repl_fn(m, z_draws(index)));
            if replace
                last_replacement = t;
            end
                
            if t > burn_in  % Save the data if past the burn-in period
                data_raw(i, t-burn_in, 1) = replace;
                data_raw(i, t-burn_in, 2) = m;
                data_raw(i, t-burn_in, 3:2+size_w) = w_draws(index, :);

                if replace
                    durations = [durations; t - last_replacement];  % Time from current replacement to previous ("last last replacement" after the current one)
                end
            end
            
            %* Calculate next-period mileage based on current-period choice
            shock = abs(normrnd(0, sqrt(pi/(pi-2)))) * (1+z_draws(index));
            if replace == 1
                m = shock;
            else
                m = m + shock;
            end
        end    
    end

    data = cat(3, data_raw(:, 1:periods, :), data_raw(:, 2:periods+1, :));
end