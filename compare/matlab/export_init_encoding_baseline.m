% Export MATLAB baseline for init_sol / encoding / decoding (no algorithm loop).

this_file = mfilename('fullpath');
[this_dir, ~, ~] = fileparts(this_file);
obj_chain_dir = fileparts(this_dir);
baseline_dir = fullfile(obj_chain_dir, 'baseline_matlab');
if ~exist(baseline_dir, 'dir')
    mkdir(baseline_dir);
end

project_dir = fileparts(fileparts(obj_chain_dir)); % .../eee
addpath(project_dir);
addpath(fullfile(project_dir, 'mobka'));

% use project CSV shape for dimensions

demand_points_info = csvread(fullfile(project_dir, 'demand_points_info.csv'));
charge_points_info = csvread(fullfile(project_dir, 'charge_points_info.csv'));

charge_points_num = size(charge_points_info, 1);
demand_points_num = size(demand_points_info, 1);

seed = 2;
num_individuals = 5;

rng(seed, 'twister');

init_solutions = zeros(num_individuals, charge_points_num);
active_count = zeros(num_individuals, 1);

active_idx_cell = cell(num_individuals, 1);
active_val_cell = cell(num_individuals, 1);

for i = 1:num_individuals
    sol = init_sol(charge_points_num, demand_points_num);
    init_solutions(i, :) = sol;

    idx = find(sol > 0);  % 1-based index
    vals = sol(idx);

    active_count(i) = numel(idx);
    active_idx_cell{i} = idx;
    active_val_cell{i} = vals;
end

max_active = max(active_count);
active_idx_pad = -1 * ones(num_individuals, max_active);
active_val_pad = nan(num_individuals, max_active);

for i = 1:num_individuals
    c = active_count(i);
    if c > 0
        active_idx_pad(i, 1:c) = active_idx_cell{i};
        active_val_pad(i, 1:c) = active_val_cell{i};
    end
end

writematrix([seed], fullfile(baseline_dir, 'init_seed.csv'));
writematrix([charge_points_num], fullfile(baseline_dir, 'init_charge_points_num.csv'));
writematrix([demand_points_num], fullfile(baseline_dir, 'init_demand_points_num.csv'));
writematrix(init_solutions, fullfile(baseline_dir, 'mat_init_solutions.csv'));
writematrix(active_count, fullfile(baseline_dir, 'mat_init_active_count.csv'));
writematrix(active_idx_pad, fullfile(baseline_dir, 'mat_init_active_idx_pad.csv'));
writematrix(active_val_pad, fullfile(baseline_dir, 'mat_init_active_val_pad.csv'));

save(fullfile(baseline_dir, 'mat_init_debug.mat'), 'seed', 'charge_points_num', 'demand_points_num', 'init_solutions', 'active_count', 'active_idx_cell', 'active_val_cell', 'active_idx_pad', 'active_val_pad');

disp('Init baseline exported to:');
disp(baseline_dir);
