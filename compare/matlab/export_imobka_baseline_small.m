% Export IMOBKA small baseline.

this_file = mfilename('fullpath');
[this_dir, ~, ~] = fileparts(this_file);
obj_chain_dir = fileparts(this_dir);
baseline_dir = fullfile(obj_chain_dir, 'baseline_matlab');
if ~exist(baseline_dir, 'dir')
    mkdir(baseline_dir);
end

project_dir = fileparts(fileparts(obj_chain_dir));
addpath(project_dir);
addpath(fullfile(project_dir, 'mobka'));

settings.data1 = fullfile(project_dir, 'demand_points_info_10.csv');
settings.data2 = fullfile(project_dir, 'charge_points_info_10.csv');
settings.maxgen = 8;
settings.popnum = 20;

rng(2, 'twister');

% Build initial population same as runme

demand_points_info = csvread(settings.data1);
charge_points_info = csvread(settings.data2);
[demand_points_num, ~] = size(demand_points_info);
[charge_points_num, ~] = size(charge_points_info);
parameter = get_parameter();

obj_manager = [];
sol_manager = {};
for i = 1:settings.popnum
    curr_sol = init_sol(charge_points_num, demand_points_num);
    [obj_1, obj_2] = cal_obj_1(curr_sol, demand_points_info, charge_points_info, parameter);
    while isequal([obj_1, obj_2], [inf, inf])
        curr_sol = init_sol(charge_points_num, demand_points_num);
        [obj_1, obj_2] = cal_obj_1(curr_sol, demand_points_info, charge_points_info, parameter);
    end
    obj_manager = [obj_manager; [obj_1, obj_2]];
    sol_manager{end + 1} = curr_sol;
end
settings.obj_manager = obj_manager;
settings.sol_manager = sol_manager;

try
    final_obj = IMOBKA_funciton(settings);
catch
    % In some source snapshots IMOBKA_funciton.m primary function is named MOBKA_funciton.
    final_obj = MOBKA_funciton(settings);
end

writematrix(final_obj, fullfile(baseline_dir, 'mat_imobka_small_final_obj.csv'));
save(fullfile(baseline_dir, 'mat_imobka_small_debug.mat'), 'settings', 'final_obj');

disp('IMOBKA baseline exported.');
