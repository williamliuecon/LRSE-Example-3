% This code creates tables of the results from estimatenew.m, which correspond to different sample sizes and choices of basis functions.
%
% AUTHOR
% William Liu (liuw@mit.edu) 2024

%% Code parameters
clear
spec = "1"  % string variable: 1, 1.5, or int >= 2
foldername = "Rule of Thumb"
% foldername = "Cross-validation"  % string variable: "Rule of Thumb" or "Cross-validation"

n_array = ["n=100", "n=300", "n=1000", "n=10000"];
% n_array = ["n=100", "n=300", "n=1000"];

%% Make table
[meanbias, meansd, truesd, medianbias, mediansd, coverageprob] = deal(nan(6,2));
rownames = []
for i = 1:length(n_array)
    rows = 2*i-1:2*i;
    n_str = n_array(i);
    
    %* Load data
    filename = strcat("est_t=10_", n_str, "_", spec, ".mat");
    load(fullfile(foldername, filename));
    
    %* Calculate results
    [meanbias(rows, :), meansd(rows, :), truesd(rows, :), ...
        medianbias(rows, :), mediansd(rows, :), coverageprob(rows, :)] = get_results(est, sd);
    
    %* Make row names
    rownames = [rownames, strcat("θ_1 (", n_str, ")"), strcat("θ_2 (", n_str, ")")];
end

colnames = ["PI Bias", "DB Bias", "Median SE", "PI SD", "DB SD", "PI Coverage", "DB Coverage"];
T = table(meanbias(:, 1), meanbias(:, 2), mediansd(:, 2), ...
    truesd(:, 1), truesd(:, 2), coverageprob(:, 1), coverageprob(:, 2), ...
    'VariableNames', colnames, 'RowNames', rownames);

%* Rounding to three decimal places
new_T = varfun(@(x) num2str(x, ['%' sprintf('.%df', 3)]), T);
new_T.Properties.VariableNames = T.Properties.VariableNames;
new_T.Properties.RowNames = T.Properties.RowNames;
new_T

%* Saving
% writetable(new_T, "results.csv", "WriteRowNames", true);



%* Function to calculate results
function [meanbias, meansd, truesd, medianbias, mediansd, coverageprob] = get_results(est, sd)
    meanbias = abs(nanmean(est, 3) - [-1; -0.5]);
    meansd = nanmean(sd, 3);
    truesd = sqrt(nanvar(est, [], 3));
    medianbias = abs(median(est, 3) - [-1; -0.5]);
    mediansd = median(sd, 3);
    coverageprob = nanmean((abs(est - [-1; -0.5]) ./ sd) <= 1.96, 3);
end