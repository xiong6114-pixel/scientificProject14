% Export MATLAB NSGA-II small experiment baseline (no IMOBKA, no other main loops).

this_file = mfilename('fullpath');
[this_dir, ~, ~] = fileparts(this_file);
obj_chain_dir = fileparts(this_dir);
baseline_dir = fullfile(obj_chain_dir, 'baseline_matlab');
if ~exist(baseline_dir, 'dir')
    mkdir(baseline_dir);
end

project_dir = fileparts(fileparts(obj_chain_dir)); % .../eee
addpath(project_dir);
addpath(fullfile(project_dir, 'NSGAII'));

data1 = fullfile(project_dir, 'demand_points_info_10.csv');
data2 = fullfile(project_dir, 'charge_points_info_10.csv');

settings.maxgen = 15;
settings.popnum = 30;
settings.data1 = data1;
settings.data2 = data2;

seed = 2;
rng(seed, 'twister');

% Build initial population as in runme initialization logic.
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

% NSGA-II loop (same as NSGA2_funciton.m) + generation metrics.
max_iter_num = settings.maxgen;
pop_num = settings.popnum;
variate_rate = 0.3;
cross_rate = 0.8;

generation_metrics = [];
for iter_num = 1:max_iter_num
    for i = 1:pop_num
        if rand() < variate_rate
            new_sol = variate(sol_manager{i}, charge_points_num, demand_points_num);
            [obj_1, obj_2] = cal_obj(new_sol, demand_points_info, charge_points_info, parameter);
            if obj_1 ~= -1 || obj_2 ~= -1
                obj_manager = [obj_manager; [obj_1, obj_2]];
                sol_manager{end + 1} = new_sol;
            end
        end
        if rand() < cross_rate
            rand_id = randi([1, pop_num]);
            new_sol = cross(sol_manager{i}, charge_points_num, sol_manager{rand_id});
            [obj_1, obj_2] = cal_obj(new_sol, demand_points_info, charge_points_info, parameter);
            if obj_1 ~= -1 || obj_2 ~= -1
                obj_manager = [obj_manager; [obj_1, obj_2]];
                sol_manager{end + 1} = new_sol;
            end
        end
    end

    [fronts_tmp, ~] = fast_non_dominated_sort_with_crowding(obj_manager);
    nd_count = length(fronts_tmp{1});
    best_obj1 = min(obj_manager(:,1));
    best_obj2 = min(obj_manager(:,2));
    mean_obj1 = mean(obj_manager(:,1));
    mean_obj2 = mean(obj_manager(:,2));
    generation_metrics = [generation_metrics; [iter_num, best_obj1, best_obj2, mean_obj1, mean_obj2, nd_count]];

    if iter_num == max_iter_num
        break
    end

    [fronts, crowding_dist] = fast_non_dominated_sort_with_crowding(obj_manager);
    new_sol_manager = {};
    new_obj_manager = [];
    for ii = 1:length(fronts)
        list = fronts{ii};
        crowd = [];
        recorder = [];
        for jj = 1:length(list)
            sol_id = list(jj);
            recorder(end + 1) = sol_id;
            crowd(end + 1) = crowding_dist(sol_id);
        end
        [~, idx] = sort(crowd, 'descend');
        for jj = 1:length(list)
            curr_sol_id = recorder(idx(jj));
            new_sol_manager{end + 1} = sol_manager{curr_sol_id};
            new_obj_manager = [new_obj_manager; [obj_manager(curr_sol_id, 1), obj_manager(curr_sol_id, 2)]];
        end
    end

    sol_manager = {};
    obj_manager = [];
    for ii = 1:pop_num
        sol_manager{end + 1} = new_sol_manager{ii};
        obj_manager = [obj_manager; [new_obj_manager(ii, 1), new_obj_manager(ii, 2)]];
    end
end

[fronts, ~] = fast_non_dominated_sort_with_crowding(obj_manager);
final_obj = [];
list = fronts{1};
for j = 1:length(list)
    if j > 1
        sol_id = list(j);
        find_flag = 0;
        [num, ~] = size(final_obj);
        for k = 1:num
            if obj_manager(sol_id, 1) == final_obj(k, 1) && obj_manager(sol_id, 2) == final_obj(k, 2)
                find_flag = 1;
                break
            end
        end
        if find_flag == 1
            continue
        end
    end
    sol_id = list(j);
    final_obj = [final_obj; [obj_manager(sol_id, 1), obj_manager(sol_id, 2)]];
end

[~, sort_idx] = sort(final_obj(:,1));
front_sorted = final_obj(sort_idx, :);

writematrix([seed], fullfile(baseline_dir, 'nsga2_small_seed.csv'));
writematrix([settings.maxgen], fullfile(baseline_dir, 'nsga2_small_maxgen.csv'));
writematrix([settings.popnum], fullfile(baseline_dir, 'nsga2_small_popnum.csv'));
writematrix(final_obj, fullfile(baseline_dir, 'mat_nsga2_small_final_obj.csv'));
writematrix(front_sorted, fullfile(baseline_dir, 'mat_nsga2_small_front_sorted.csv'));
writematrix(generation_metrics, fullfile(baseline_dir, 'mat_nsga2_small_generation_metrics.csv'));

save(fullfile(baseline_dir, 'mat_nsga2_small_debug.mat'), 'seed', 'settings', 'final_obj', 'front_sorted', 'generation_metrics');

disp('NSGA-II small baseline exported to:');
disp(baseline_dir);
