% Export NSGA3 small baseline.

this_file = mfilename('fullpath');
[this_dir, ~, ~] = fileparts(this_file);
obj_chain_dir = fileparts(this_dir);
baseline_dir = fullfile(obj_chain_dir, 'baseline_matlab');
if ~exist(baseline_dir, 'dir')
    mkdir(baseline_dir);
end

project_dir = fileparts(fileparts(obj_chain_dir));
addpath(project_dir);
addpath(fullfile(project_dir, 'NSGAIII'));

settings.data1 = fullfile(project_dir, 'demand_points_info_10.csv');
settings.data2 = fullfile(project_dir, 'charge_points_info_10.csv');
settings.maxgen = 8;
settings.popnum = 20;

rng(2, 'twister');

final_obj = NSGA3_funciton(settings);
writematrix(final_obj, fullfile(baseline_dir, 'mat_nsga3_small_final_obj.csv'));
save(fullfile(baseline_dir, 'mat_nsga3_small_debug.mat'), 'settings', 'final_obj');

disp('NSGA3 baseline exported.');
