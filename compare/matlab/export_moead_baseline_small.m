% Export MOEAD small baseline.

this_file = mfilename('fullpath');
[this_dir, ~, ~] = fileparts(this_file);
obj_chain_dir = fileparts(this_dir);
baseline_dir = fullfile(obj_chain_dir, 'baseline_matlab');
if ~exist(baseline_dir, 'dir')
    mkdir(baseline_dir);
end

project_dir = fileparts(fileparts(obj_chain_dir));
addpath(project_dir);
addpath(fullfile(project_dir, 'moead'));

settings.data1 = fullfile(project_dir, 'demand_points_info_10.csv');
settings.data2 = fullfile(project_dir, 'charge_points_info_10.csv');
settings.maxgen = 8;
settings.popnum = 20;

rng(2, 'twister');

% Build initial population as in runme-style init

demand_points_info = csvread(settings.data1);
charge_points_info = csvread(settings.data2);
[demand_points_num, ~] = size(demand_points_info);
[charge_points_num, ~] = size(charge_points_info);
parameter = get_parameter();

obj_manager = [];
sol_manager = {};
for i = 1:settings.popnum
    curr_sol = init_sol(charge_points_num, demand_points_num);
    [obj_1, obj_2] = cal_obj(curr_sol, demand_points_info, charge_points_info, parameter);
    while obj_1 == -1 && obj_2 == -1
        curr_sol = init_sol(charge_points_num, demand_points_num);
        [obj_1, obj_2] = cal_obj(curr_sol, demand_points_info, charge_points_info, parameter);
    end
    obj_manager = [obj_manager; [obj_1, obj_2]];
    sol_manager{end + 1} = curr_sol;
end
settings.obj_manager = obj_manager;
settings.sol_manager = sol_manager;

final_obj = MOEAD_function(settings);
writematrix(final_obj, fullfile(baseline_dir, 'mat_moead_small_final_obj.csv'));

save(fullfile(baseline_dir, 'mat_moead_small_debug.mat'), 'settings', 'final_obj');

disp('MOEAD baseline exported.');
