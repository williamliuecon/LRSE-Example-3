% This function returns the point estimates and standard errors in two matrices.
% The first column has the raw estimates and the second column the bias-corrected estimates.
%
% AUTHOR
% William Liu (liuw@mit.edu) 2024

%% Main function
function [estimates, standard_errors] = estimate_fn(Data, I, size_x, delta, spec, rule_of_thumb)
    % INPUTS
    % Data – dataset
    % I – indexes the sub-samples for sample splitting
    % size_x – the number of all state variables
    % delta – discount factor
    % spec – determines what basis functions to use
    
    %% Setup
    %* Code Parameters
    fsolve_opt = optimset('Display', 'off');  % Disables display output from fsolve()
    
    %* Define functions
    Lambda = @(u) 1 ./ (1+(exp(-u)));  % Standard logistic CDF
    dLambda = @(u) exp(-u) .* (Lambda(u).^2);  % Standard logistic PDF; Lambda_a in the paper
    % pi_fn = @(u) dLambda(u) ./ (Lambda(u) .* (1-Lambda(u)));  % pi_fn(a_hat) equals 1 in Example 3, so it is left out here
    H = @(p) -log(1-p);  % Omit constant (Euler's constant) because it doesn't matter
    H_p = @(p) 1 ./ (1-p);  % Derivative of H
    
    N = size(Data, 1);
    k_folds = size(I, 2);
    ones_N = ones(N, 1);
    ones_N_div_k = ones(N/k_folds, 1);

    %* Get variables from columns in Data (for readability)
    y_1 = Data(:, 1);
    y_2 = 1 - y_1;
    x = Data(:, 2:1+size_x);
    y_n = 1 - Data(:, 2+size_x);
    x_n = Data(:, 3+size_x:end);
    
    %* Generate basis functions
    if spec == 1
        X = x;
        X_n = x_n;
    elseif spec == 1.5
        X = [x, x.^2];
        X_n = [x_n, x_n.^2];

        X = unique(X.', 'rows', 'stable').';  % Remove duplicate columns (inefficient but easy code)
        X_n = unique(X_n.', 'rows', 'stable').';
    elseif spec >= 2
        model = allVL1(size(x, 2), spec);
        X = x2fx(x, model);
        X_n = x2fx(x_n, model);

        X = unique(X.', 'rows', 'stable').';  % Remove duplicate columns (inefficient but easy code)
        X_n = unique(X_n.', 'rows', 'stable').';
    end
    
    %% Estimate gammas
    %* Initialize the objects
    g_1_hat_init = nan(N, k_folds);
    g_1_hat = nan(N, 1);  % gamma_1_hat(X_t+1)
    g_2_hat = nan(N, 1);  % gamma_2_hat(X_t)
    g_3_hat = nan(N, 1);  % gamma_3_hat -- this is not a function
    
    for i = 1:k_folds        
        %* Estimate gamma_1_ll' (two-way sample splitting)
        for j = setdiff(1:k_folds, i)  % Want all possible combinations of (i, j) with i != j
            comp = logical(ones_N - I(:, i) - I(:, j));
            coef = get_lassoglm_coef(X(comp, :), y_2(comp), rule_of_thumb);
            g_1_hat_init(I(:, j), i) = glmval(coef, X_n(I(:, j), :), 'logit');  % Evaluation at X_t+1
            g_1_hat_init(I(:, j), i) = min(max(g_1_hat_init(I(:, j), i), 0.01), 0.99);
            % Force the elements of g_1_hat to be in [0.01, 0.99].
            % The right endpoint is to prevent H(g) from blowing up for g close to 1. The left endpoint is for symmetry.
        end
        
        %* This bunch of indices will be used for the next three estimators
        comp = logical(ones_N - I(:, i));
        
        %* Estimate gamma_1_l (one-way sample splitting)
        coef = get_lassoglm_coef(X(comp, :), y_2(comp), rule_of_thumb);
        g_1_hat(I(:, i)) = glmval(coef, X_n(I(:, i), :), 'logit');  % Evaluation at X_t+1
        g_1_hat(I(:, i)) = min(max(g_1_hat(I(:, i), :), 0.01), 0.99);
        
        %* Estimate gamma_2_l
        tempx = y_2(comp) .* X(comp, :);
        H_g_1_hat_init_comp = H(g_1_hat_init(comp, i));  % Uses a subset of the vector gamma_1_hat_ll'
        ycomp_nonzero_filt = logical(y_2(comp));
        coef = get_lasso_coef(tempx(ycomp_nonzero_filt, :), H_g_1_hat_init_comp(ycomp_nonzero_filt), rule_of_thumb);
        g_2_hat(I(:, i)) = [ones_N_div_k, X(I(:, i), :)] * coef;  % Evaluation at X_t
        
        %* Estimate gamma_3_l
        g_3_hat(I(:, i)) = mean(y_1(comp) .* H_g_1_hat_init_comp) / mean(y_1(comp));
        % Could be made to be more efficient by directly conditioning on Y_1 = 1 and then summing; though, both are equivalent
    end
    
    %% Estimate theta_tilde_l and construct a_hat
    V_diff = delta * (g_2_hat - g_3_hat);  % The expected difference between the value fns for the two options, (3.3)
    D = [-ones_N, sqrt(1+x(:, 1))];
    a = @(theta) D * theta + V_diff;
    mom = @(theta) D .* repmat(y_2 - Lambda(a(theta)), [1, 2]);  % Nx2, with each mom. fn contribution being 1x2; pi_fn omitted
    a_subset = @(theta, rows) D(rows, :) * theta + V_diff(rows);
    mom_subset = @(theta, rows) D(rows, :) .* repmat(y_2(rows) - Lambda(a_subset(theta, rows)), [1, 2]);  % pi_fn omitted
    % Have to make a separate anon. fn for subsetting because opt. arg not allowed for anon. fn

    % theta_tilde_l is not necessarily a scalar, so we can't just stack them to get theta_tilde and then do a_hat = a(theta_tilde)
    a_hat = nan(N, 1);
    
    for i = 1:k_folds
        comp = logical(ones_N - I(:, i));
        fun = @(theta) sum(mom_subset(theta, comp)).';
        theta_tilde_l = fsolve(fun, [0; 0], fsolve_opt);
        % 2x1 moment function for 2x1 theta; system is square and just-identified
        % fsolve() will also understand if the moment function is not transposed and is left as 1x2, but it will take longer
        a_hat(I(:, i)) = a_subset(theta_tilde_l, I(:, i));
    end
    
    %% Construct alpha_hat
    % alpha_..._hat will refer to the stacked version of alpha_..._hat_l.
    % Also, note that the phis and alphas all have two components.
    dLambda_a_hat = dLambda(a_hat);

    %* Estimate alpha_2_hat
    alpha_2_hat = -delta * D .* dLambda_a_hat .* y_2 ./ Lambda(a_hat);  % pi_fn omitted
    
    part1 = nan(N, 2);
    part2 = nan(N, 1);
    alpha_3_hat = nan(N, 2);

    for i = 1:k_folds
        comp = logical(ones_N - I(:, i));

        %* Estimate alpha_3_hat using alpha_2_hat
        alpha_3_hat(I(:, i), :) = -mean(alpha_2_hat(comp)) ./ mean(y_1(comp));  % Note that this varies only across folds
        
        %* Estimate alpha_1_hat using alpha_2_hat and alpha_3_hat
        coef = get_lasso_coef(X_n(comp, :), alpha_2_hat(comp, 1), rule_of_thumb);
        part1(I(:, i), 1) = [ones_N_div_k, X_n(I(:, i), :)] * coef;  % Evaluation at X_t+1
        coef = get_lasso_coef(X_n(comp, :), alpha_2_hat(comp, 2), rule_of_thumb);
        part1(I(:, i), 2) = [ones_N_div_k, X_n(I(:, i), :)] * coef;  % Evaluation at X_t+1
        coef = get_lasso_coef(X_n(comp, :), y_1(comp), rule_of_thumb);
        part2(I(:, i)) = [ones_N_div_k, X_n(I(:, i), :)] * coef;  % Evaluation at X_t+1
    end
    alpha_1_hat = (part1 + alpha_3_hat .* part2) .* H_p(g_1_hat);
        
    %% Calculate phi_hat using alpha_hat
    % We won't bother to average over t because this makes no difference in the moment function.
    H_g_1_hat = H(g_1_hat);
    phi_1_hat = alpha_1_hat .* (y_n - g_1_hat);
    phi_2_hat = alpha_2_hat .* (H_g_1_hat - g_2_hat);
    phi_3_hat = alpha_3_hat .* y_1 .* (H_g_1_hat - g_3_hat);
        
    %% Estimate theta with and without bias correction
    %* Uncorrected estimate
    % The sample splitting enters through g_2_hat and g_3_hat in a_hat.
    fun = @(theta) sum(mom(theta)).';
    est1 = fsolve(fun, [0; 0], fsolve_opt);
    
    %* Corrected estimate using phi_1, phi_2, and phi_3
    % The sample splitting enters through g_2_hat and g_3_hat in a_hat, as well as phi_1_hat, phi_2_hat, and phi_3_hat.
    mom_corr = @(theta) mom(theta) + phi_1_hat + phi_2_hat + phi_3_hat;
    fun = @(theta) sum(mom_corr(theta)).';
    est4 = fsolve(fun, est1, fsolve_opt);
    
    %% Calculate standard errors
    % This is done for all the moment function variants, using the numerical derivative of each.
    G = D.' * bsxfun(@times, D, -dLambda_a_hat);  % Efficiently calculates D' * diag(-dLambda_a_hat) * D
    G_inv = inv(G);  % Would be more efficient to instead use LU decomposition to solve for V
    % D and D' are swapped because D is 2x1 in the paper but is 1x2 here.

    mom_1_eval = mom(est1);
    Omega = mom_1_eval.' * mom_1_eval;
    covar = (G_inv * Omega * G_inv.');
    sd1 = sqrt(diag(covar));
    
    mom_4_eval = mom_corr(est4);
    Omega_corr = mom_4_eval.' * mom_4_eval;
    covar = (G_inv * Omega_corr * G_inv.');
    sd4 = sqrt(diag(covar));
    
    %% Return the results
    estimates = [est1, est4];
    standard_errors = [sd1, sd4];
end



%% Auxiliary functions
%* Runs the logit lasso estimation and gets the coefficients
function [coef] = get_lassoglm_coef(X, y, rule_of_thumb)
    if rule_of_thumb
        rule_of_thumb_penalty = norminv(1 - 0.1 / (2*size(X, 2))) / size(y, 1);
        [params, fitinfo] = lassoglm(X, y, 'binomial', 'Lambda', rule_of_thumb_penalty);
        coef = [fitinfo.Intercept; params];
    else
        [params, fitinfo] = lassoglm(X, y, 'binomial', 'CV', 2);
        idx = fitinfo.IndexMinDeviance;
        coef = [fitinfo.Intercept(idx); params(:, idx)];
    end
end

%* Runs the linear lasso estimation and gets the coefficients
function [coef] = get_lasso_coef(X, y, rule_of_thumb)
    if rule_of_thumb
        rule_of_thumb_penalty = norminv(1 - 0.1 / (2*size(X, 2))) / size(y, 1);
        [params, fitinfo] = lasso(X, y, 'Lambda', rule_of_thumb_penalty);
        coef = [fitinfo.Intercept; params];
    else
        [params, fitinfo] = lasso(X, y, 'CV', 2);
        idx = fitinfo.IndexMinMSE;
        coef = [fitinfo.Intercept(idx); params(:, idx)];
    end
end