function [integ_V_fn, static_action_V_main, static_action_V_repl, maintenance_EV, replacement_EV] = solve_Markov_DP_fn(params, m_grid, z_grid, draws, tol, method, init_fn)
    % The arguments:
    % * params – struct containing the regression single-index parameters "c_k" as well as other parameters
    % * m_grid – grid of values for the persistent state variable (mileage)
    % * z_grid – grid of values for the regression single-index SUM_{k} c_k * X_t,k+1
    % * draws – number of Monte Carlo draws used to approximate the (conditional) expected value function for each action
    % * tol – tolerance threshold for convergence of integrated value function in the value iteration loop
    % * method – the interpolation method used for interp2() calls
    % * init_fn – initial guess for the integrated value function in the value iteration loop

    %% Preliminaries
    size_w = size(params.coeff_w, 2);  % Number of iid state variables that affect transitions
    size_m_grid = size(m_grid, 2);
    size_z_grid = size(z_grid, 2);
    [m_meshgrid, z_meshgrid] = meshgrid(m_grid, z_grid);
    rng("default");  % Set seed
    
    %% Monte Carlo simulation of next-period variables
    %* Simulate next-period persistent state variable (mileage) for each action, taking draws for each (m, z) point
    % Maintenance
    shock_main = abs(normrnd(0, sqrt(pi/(pi-2)), size_z_grid, size_m_grid, draws)) .* repmat(1+z_grid', 1, size_m_grid, draws);  % Half-normal with variance 1, times 1+z
    m_new_draws_main = repmat(m_grid, size_z_grid, 1, draws) + shock_main;
    
    % Replacement
    % * Note that the replacement action-value function is a constant that does not depend on m or z.
    %   Consequently, it makes sense to average over all of the (m, z) points (or rewrite that section of code entirely, but that's pointless effort).
    %   Because of this averaging, there will be draws_repl * size_z_grid * size_m_grid draws used for the replacement route.
    %   Therefore, draws_repl can be made smaller to save on computational cost.
    % * m_new_draws_repl is "shock_repl"; it is essentially the same as shock_main but with fewer Monte Carlo draws.
    draws_repl = ceil(draws / (size_z_grid * size_m_grid));  % Smallest integer such that draws_repl * size_z_grid * size_m_grid >= draws
    m_new_draws_repl = abs(normrnd(0, sqrt(pi/(pi-2)), size_z_grid, size_m_grid, draws_repl)) .* repmat(1+z_grid', 1, size_m_grid, draws_repl);
    
    %* Simulate next-period iid state variables 
    % The iid state variables are each Uniform or Bernoulli. They are arranged in alternating order:
    %     Uniform, Bernoulli, Uniform, Bernoulli, Uniform, ...
    w_new_unif = sqrt(12) * rand(draws, ceil(size_w/2));
    w_new_disc = binornd(1, 0.5, draws, floor(size_w/2));
    if mod(size_w, 2) ~= 0  % Interleave columns and then tranpose
        w_new_draws = cat(2, reshape([w_new_unif(:, 1:end-1); w_new_disc], size(w_new_unif, 1), []), w_new_unif(:, end));
    else
        w_new_draws = reshape([w_new_unif(:, 1:end); w_new_disc], size(w_new_unif, 1), []);
    end
    z_new_draws = repmat(reshape(params.coeff_w * w_new_draws', 1, 1, draws), size_z_grid, size_m_grid, 1);
    
    %% Value iteration to calculate action-value functions
    %* Initialize integrated value function
    if nargin < 7
        integ_V_old = zeros(size_z_grid, size_m_grid);
    else
        integ_V_old = init_fn(m_meshgrid, z_meshgrid);
    end
    integ_V_change = Inf;  % Initialize the variable that stores the max abs difference between value function iterations
    count = 0;
    while 1
        %* Given previous value function iteration, calculate draws for the expected value function for each action
        % Here, -Inf is used as the extrapval to "disallow" extrapolation rather than NaN.
        % When NaN is used as the extrapval here, computational issues arise (an infinite loop).
        % Note: extrapval is NaN by default if the method is not 'spline' or 'makima'.
        maintenance_EV_draws = interp2(m_meshgrid, z_meshgrid, integ_V_old, m_new_draws_main, z_new_draws, method, -Inf);
        replacement_EV_draws = interp2(m_meshgrid, z_meshgrid, integ_V_old, m_new_draws_repl, z_new_draws(:, :, 1:draws_repl), method, -Inf);
        
        %* Exit pseudo-infinite loop when tolerance level reached
        if integ_V_change <= tol
            break
        end
        
        %* Calculate expected value function for each action by averaging over the draws
        maintenance_EV = mean(maintenance_EV_draws, 3);      % matrix
        replacement_EV = mean(replacement_EV_draws, "all");  % scalar (see earlier comment)

        %* Calculate new iteration of the (static component of the) action-value function for each action
        % * "static_action_V_main" has a cross-sectional shape in the "m" dimension similar to sqrt(a), as expected.
        %   However, it only decreases slightly and then flattens out in the "z" dimension.
        %   This is because, whilst a high z causes a high next-period mileage, the agent can just choose replacement next period to avoid being "hit" by it.
        %   (On the other hand, the shape near the origin is determined by how z appears in the shock.)
        % * Use sqrt(1+m) to avoid infinite gradient and get better numerical approximation
        static_action_V_main = params.maintenance_factor * sqrt(1+m_meshgrid) + params.beta * maintenance_EV;
        static_action_V_repl = params.replacement_cost + params.beta * replacement_EV;
        
        %* Calculate new value function iteration
        integ_V = log(exp(static_action_V_main) + exp(static_action_V_repl));
        integ_V_change = max(abs(integ_V - integ_V_old), [], "all");
        integ_V_old = integ_V;
        count = count + 1;
    end
    fprintf("Number of iterations: %d\n", count)

    %* Return version of integ_V that can be used as initfn
    integ_V_fn = @(m, z) interp2(m_meshgrid, z_meshgrid, integ_V, m, z, method, -Inf);
end