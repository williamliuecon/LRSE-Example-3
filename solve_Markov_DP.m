% This file runs "solve_Markov_DP_fn.m", which uses value iteration to solve the Markov decision problem, returning the conditional probability of replacement.
% Because the per-period utility function is random (namely, it additively contains a random heterogeneity term), we actually solve the "integrated Bellman equation,"
% where "integrated" just means transforming the Bellman equation by averaging over the distribution of the heterogeneity term.
% 
% This explanation will use the terminology of Aguirregabiria and Mira (2010) and similar notation.
% Subscript i will be suppressed. j will denote the choice made in the current time period.
% Let the integrated value function V_bar denote the expectation of the value function w.r.t. to the utility function heterogeneity,
% let the expected value function EV denote the expectation of the value function w.r.t. to both the utility function heterogeneity and
% the state variable at time t+1, X_1,t+1, and let nu denote the static component of the action-value function,
% which in this example is a static utility term plus the discounted next-period expected value function for the given action.
% "State variable X_t" is just used as shorthand here for the mileage and the iid variables.
% 
% We simply do the "inner loop" of Rust's (1987) nested fixed point algorithm to solve for nu.
% This algorithm is a loop whose steps are as follows:
% (0) First, note that we are given X_t and the previous iteration of the integrated value function, V_bar.
%     X_t is taken from a grid of points. On the other hand, V_bar is calculated iteratively – the first iteration is just be V_bar = 0 everywhere.
% (1) For each choice, using X_t, we create Monte Carlo draws for X_t+1.
% (2) For each choice, we plug in these X_t+1 draws into V_bar.
%     (Technically, we have only evaluated V_bar at the points of the grid for X_t, so we actually need to interpolate when we do this.)
% (3) For each choice, taking the mean of V_bar(X_t+1) over the Monte Carlo draws will give us EV_new(X_t+1, j).
%     This denotes the new iteration of the expected value function, EV_new, evaluated at time t+1 and at choice j.
% (4) We then calculate the new iteration of the integrated value function, V_bar_new, by simply plugging EV_new(X_t+1, j) into a formula:
%         V_bar_new(X_t) = ln{ SUM_j { exp( nu(X_t, j) ) } },
%     where nu(X_t, j) = u(X_t, j) + β*EV_new(X_t+1, j) is the static component of the action-value function for action j
%     and u() is the static utility term of the utility function.
% (5) We replace V_bar with V_bar_new and repeat the loop until V_bar and V_bar_new are sufficiently similar.
% 
% Then, we just need to plug nu(X_t, j) into a formula for the conditional choice probabilities.
% In this example, it is asssumed that the heterogeneity in the random per-period utility function has a Type I extreme value distribution.
% Consequently, the formula will have a logit form.
% (Since there are only two possible actions, we actually just need to directly calculate the conditional probability for only one of them.)
% 
% AUTHOR
% William Liu (liuw@mit.edu) 2024

%% Preliminaries
clear

%* Create params struct
size_w = 4;  % Number of iid state variables that affect transitions
params.maintenance_factor = -0.5;  % Per-period maintenance cost coefficient
params.replacement_cost = -1;  % Replacement cost
params.beta = 0.9;  % Discount factor
params.coeff_w = 0.1 * (1:size_w).^-2;  % Coefficients c_k

tol = 0.000001;  % Easy to show that tolerance threshold for cond_prob_repl is bounded above by 0.45 * tolerance threshold for integ_V.

%% Do value iteration
% * Make initial guess with coarse grid
% * Unfortunately, using the interp2() interpolation method "cubic" cannot be used here because m_grid is not uniformly spaced.
%   When there are many points and a non-uniform grid, using "spline" causes a "Insufficient finite values to interpolate"
%   error when NaN values are used as the "extrapval" rather 1, so "linear" should be used during testing with NaN values.
% * "solve_Markov_DP_fn()" is in its own file so that this file can be run on a HPC cluster.
% * m and z have a minimum possible value of 0
% * In the m dimension, the extrapolation jump to 1 occurs at around >0.9999; m_grid is taken large enough so that the jump is small.
% * z has a maximum possible value of sqrt(12)/10 + 1/40 + sqrt(12)/90 + 1/160 < 0.4162.
% * It is important to make sure that all of the MC z draws are within z_grid and that we do not extrapolate in the z dimension!
%   This is because the estimated conditional probability of replacement jumps to 1 when extrapolating.
%   (In the z dimension with m=0, the conditional probability of replacement never approaches 1 as z goes to infinity because,
%   whilst a high z causes a high next-period mileage, the agent can just choose replacement next period to avoid being "hit" by it.)
% * If f is an approximation of the shape of cond_prob_repl_fn in the m dimension (and it is assumed to not vary much by z),
%   then m_grid should be f_inv(linspace(f(start), f(end))) for equal spacing in the output (probability) dimension.
%   I use quadratic spacing for m_grid because the surface has an approx. sqrt-rate shape in m dimension for low m.
tic
draws = 10000;  % Number of draws used to simulate state variables m (mileage; persistent) and w (iid)
m_grid = linspace(0, sqrt(400), 1000).^2;
z_grid = linspace(0, 0.4162, 100);
integ_V_fn = solve_Markov_DP_fn(params, m_grid, z_grid, draws, tol, "spline");
toc

%* Make final guess with fine grid, initializing using initial guess
tic
draws = 25000;
m_grid = linspace(0, sqrt(400), 20000).^2;
z_grid = linspace(0, 0.4162, 100);
[integ_V_fn, static_action_V_main, static_action_V_repl] = solve_Markov_DP_fn(params, m_grid, z_grid, draws, tol, "spline", integ_V_fn);
toc

%% Calculate conditional choice probability of replacement
% For testing: the replacement of "incorrect" values of the numerical approximation for the conditional probability of replacement with NaN
% occurs here at the end rather than earlier to avoid the infinite loop computational issue mentioned in "do_value_it.m".
[m_meshgrid, z_meshgrid] = meshgrid(m_grid, z_grid);
cond_prob_repl = 1 ./ (1 + exp(static_action_V_main - static_action_V_repl));
% cond_prob_repl(cond_prob_repl == 1) = NaN;
cond_prob_repl_fn = @(m, z) interp2(m_meshgrid, z_meshgrid, cond_prob_repl, m, z, "spline", 1);

save("cond_prob_repl_fn.mat", "cond_prob_repl_fn", "-v7.3")

%% Check conditional probability of replacement surface to see if grids are large and fine enough
%{
load("cond_prob_repl_fn.mat")
out = cond_prob_repl_fn(m_meshgrid, z_meshgrid);
surf(m_meshgrid, z_meshgrid, out, "FaceColor", "interp");
xlabel("Mileage")
ylabel("Z")
%}