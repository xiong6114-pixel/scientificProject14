% Export MATLAB baseline for fixed-sol objective chain (cal_obj_1 + dependencies).
% This script does NOT run any optimization loop.

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

% Inputs: same CSV used by project
% csvread is kept for compatibility with original code style.
demand_points_info = csvread(fullfile(project_dir, 'demand_points_info.csv'));
charge_points_info = csvread(fullfile(project_dir, 'charge_points_info.csv'));
parameter = get_parameter();

J = size(charge_points_info, 1);
sol = zeros(1, J);
if J >= 1
    sol(1) = 3;
end
if J >= 3
    sol(3) = 2;
end
if J >= 5
    sol(5) = 4;
end

obj1 = 0;
obj2 = 0;
chosen_charge_points = [];
for i = 1:length(sol)
    if sol(i) > 0
        chosen_charge_points(end + 1) = i;
    end
end

[demand_points_num, ~] = size(demand_points_info);
dispatch = [];
assignment_matrix = zeros(demand_points_num, length(sol));

for i = 1:demand_points_num
    best_dist = 99999999;
    best_charge_point = -1;
    for j = 1:length(chosen_charge_points)
        chosen_point_id = chosen_charge_points(j);
        curr_dist = get_dist(demand_points_info(i,1), demand_points_info(i,2), charge_points_info(chosen_point_id,1), charge_points_info(chosen_point_id,2));
        if curr_dist < best_dist && curr_dist <= parameter(5)
            best_dist = curr_dist;
            best_charge_point = chosen_point_id;
        end
    end
    if best_charge_point == -1
        f = [inf; inf];
        dispatch = zeros(1, demand_points_num);
        E_j = sum(assignment_matrix, 1);
        beta_j = zeros(1, length(sol));
        P0 = zeros(1, length(sol));
        W_j = zeros(1, length(sol));

        writematrix(sol, fullfile(baseline_dir, 'input_sol.csv'));
        writematrix(demand_points_info, fullfile(baseline_dir, 'input_demand_points_info.csv'));
        writematrix(charge_points_info, fullfile(baseline_dir, 'input_charge_points_info.csv'));
        writematrix(parameter, fullfile(baseline_dir, 'input_parameter.csv'));
        writematrix(assignment_matrix, fullfile(baseline_dir, 'mat_assignment_matrix.csv'));
        writematrix(dispatch, fullfile(baseline_dir, 'mat_dispatch_1based.csv'));
        writematrix(E_j, fullfile(baseline_dir, 'mat_E_j.csv'));
        writematrix(beta_j, fullfile(baseline_dir, 'mat_beta_j.csv'));
        writematrix(P0, fullfile(baseline_dir, 'mat_P0.csv'));
        writematrix(W_j, fullfile(baseline_dir, 'mat_W_j.csv'));
        writematrix(f, fullfile(baseline_dir, 'mat_obj.csv'));
        w_j_list = W_j;
        save(fullfile(baseline_dir, 'mat_debug_dump.mat'), 'sol', 'demand_points_info', 'charge_points_info', 'parameter', 'assignment_matrix', 'dispatch', 'E_j', 'beta_j', 'P0', 'w_j_list', 'f');
        return;
    end
    dispatch(end + 1) = best_charge_point;
    assignment_matrix(i, best_charge_point) = 1;
end

E_j = sum(assignment_matrix, 1);

w_j_list = zeros(1, length(sol));
beta_j = zeros(1, length(sol));
P0 = zeros(1, length(sol));

for i = 1:length(sol)
    if sol(i) > 0
        n_j = sol(i);
        beta_tmp = 0;
        for j = 1:demand_points_num
            if dispatch(j) == i
                beta_tmp = beta_tmp + demand_points_info(j, 3);
            end
        end
        beta_j(i) = beta_tmp;

        if beta_tmp == 0
            f = [inf; inf];
            writematrix(sol, fullfile(baseline_dir, 'input_sol.csv'));
            writematrix(demand_points_info, fullfile(baseline_dir, 'input_demand_points_info.csv'));
            writematrix(charge_points_info, fullfile(baseline_dir, 'input_charge_points_info.csv'));
            writematrix(parameter, fullfile(baseline_dir, 'input_parameter.csv'));
            writematrix(assignment_matrix, fullfile(baseline_dir, 'mat_assignment_matrix.csv'));
            writematrix(dispatch, fullfile(baseline_dir, 'mat_dispatch_1based.csv'));
            writematrix(E_j, fullfile(baseline_dir, 'mat_E_j.csv'));
            writematrix(beta_j, fullfile(baseline_dir, 'mat_beta_j.csv'));
            writematrix(P0, fullfile(baseline_dir, 'mat_P0.csv'));
            writematrix(w_j_list, fullfile(baseline_dir, 'mat_W_j.csv'));
            writematrix(f, fullfile(baseline_dir, 'mat_obj.csv'));
            save(fullfile(baseline_dir, 'mat_debug_dump.mat'), 'sol', 'demand_points_info', 'charge_points_info', 'parameter', 'assignment_matrix', 'dispatch', 'E_j', 'beta_j', 'P0', 'w_j_list', 'f');
            return;
        end

        p_0 = 0;
        for k = 0:n_j - 1
            first = ((beta_tmp / parameter(2)) ^ k) / (factorial(k));
            second = (n_j * (beta_tmp / parameter(2)) ^ n_j) / (factorial(n_j) * (n_j - (beta_tmp / parameter(2))));
            p_0 = p_0 + first + second;
        end

        if p_0 <= 0
            f = [inf; inf];
            writematrix(sol, fullfile(baseline_dir, 'input_sol.csv'));
            writematrix(demand_points_info, fullfile(baseline_dir, 'input_demand_points_info.csv'));
            writematrix(charge_points_info, fullfile(baseline_dir, 'input_charge_points_info.csv'));
            writematrix(parameter, fullfile(baseline_dir, 'input_parameter.csv'));
            writematrix(assignment_matrix, fullfile(baseline_dir, 'mat_assignment_matrix.csv'));
            writematrix(dispatch, fullfile(baseline_dir, 'mat_dispatch_1based.csv'));
            writematrix(E_j, fullfile(baseline_dir, 'mat_E_j.csv'));
            writematrix(beta_j, fullfile(baseline_dir, 'mat_beta_j.csv'));
            writematrix(P0, fullfile(baseline_dir, 'mat_P0.csv'));
            writematrix(w_j_list, fullfile(baseline_dir, 'mat_W_j.csv'));
            writematrix(f, fullfile(baseline_dir, 'mat_obj.csv'));
            save(fullfile(baseline_dir, 'mat_debug_dump.mat'), 'sol', 'demand_points_info', 'charge_points_info', 'parameter', 'assignment_matrix', 'dispatch', 'E_j', 'beta_j', 'P0', 'w_j_list', 'f');
            return;
        end

        p_0 = p_0 ^ (-1);
        P0(i) = p_0;

        upper = n_j * ((beta_tmp / parameter(2)) ^ (n_j + 1));
        down = beta_tmp * factorial(n_j) * (n_j - (beta_tmp / parameter(2))) ^ 2;
        w_j = (upper / down) * p_0;
        w_j_list(i) = w_j;
    else
        w_j_list(i) = 0;
    end
end

for i = 1:demand_points_num
    chosen_point_id = dispatch(i);
    curr_dist = get_dist(demand_points_info(i,1), demand_points_info(i,2), charge_points_info(chosen_point_id,1), charge_points_info(chosen_point_id,2));
    curr_time = curr_dist / parameter(3);
    obj1 = obj1 + max(w_j_list(chosen_point_id) + curr_time - parameter(1), 0) * demand_points_info(i,3);
end

for i = 1:length(sol)
    if sol(i) > 0
        obj2 = obj2 + sol(i) * parameter(4) + charge_points_info(i, 3);
    end
end

f = [obj1; obj2];

writematrix(sol, fullfile(baseline_dir, 'input_sol.csv'));
writematrix(demand_points_info, fullfile(baseline_dir, 'input_demand_points_info.csv'));
writematrix(charge_points_info, fullfile(baseline_dir, 'input_charge_points_info.csv'));
writematrix(parameter, fullfile(baseline_dir, 'input_parameter.csv'));

writematrix(assignment_matrix, fullfile(baseline_dir, 'mat_assignment_matrix.csv'));
writematrix(dispatch, fullfile(baseline_dir, 'mat_dispatch_1based.csv'));
writematrix(E_j, fullfile(baseline_dir, 'mat_E_j.csv'));
writematrix(beta_j, fullfile(baseline_dir, 'mat_beta_j.csv'));
writematrix(P0, fullfile(baseline_dir, 'mat_P0.csv'));
writematrix(w_j_list, fullfile(baseline_dir, 'mat_W_j.csv'));
writematrix(f, fullfile(baseline_dir, 'mat_obj.csv'));
save(fullfile(baseline_dir, 'mat_debug_dump.mat'), 'sol', 'demand_points_info', 'charge_points_info', 'parameter', 'assignment_matrix', 'dispatch', 'E_j', 'beta_j', 'P0', 'w_j_list', 'f');

disp('Export finished. Files written to:');
disp(baseline_dir);
