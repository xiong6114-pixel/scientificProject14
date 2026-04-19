% Export MATLAB baseline for ZDT/HV/Coverage/plot-point modules.

this_file = mfilename('fullpath');
[this_dir, ~, ~] = fileparts(this_file);
obj_chain_dir = fileparts(this_dir);
baseline_dir = fullfile(obj_chain_dir, 'baseline_matlab');
if ~exist(baseline_dir, 'dir')
    mkdir(baseline_dir);
end

project_dir = fileparts(fileparts(obj_chain_dir)); % .../eee
addpath(fullfile(project_dir, 'mobka'));

x = zeros(1,30);
f1 = ZDT_cost(x,1);
f2 = ZDT_cost(x,2);
f3 = ZDT_cost(x,3);
x4 = zeros(1,10);
f4 = ZDT_cost(x4,4);

pf1 = ZDT_truePF(1,5);
pf4 = ZDT_truePF(4,5);

P = [0.2 0.8; 0.4 0.6];
ref = [1.0 1.0];
hv_ex = calc_hv_2d(P, ref);

A = [1 1; 2 2];
B = [1.5 1.5; 0.5 2.5];
cab = calc_coverage(A,B);
cba = calc_coverage(B,A);

% select_front_points logic from run_ZDT_IMOBKA_vs_MOBKA.m
P2 = [0.9 0.1;0.1 0.9;0.5 0.5;0.3 0.7;0.7 0.3];
P2s = sortrows(P2,1);
n = size(P2s,1);
k = 3;
idx = round(linspace(1,n,k));
idx(idx<1)=1; idx(idx>n)=n;
sel = P2s(unique(idx),:);

writematrix(f1, fullfile(baseline_dir, 'mat_zdt_f1.csv'));
writematrix(f2, fullfile(baseline_dir, 'mat_zdt_f2.csv'));
writematrix(f3, fullfile(baseline_dir, 'mat_zdt_f3.csv'));
writematrix(f4, fullfile(baseline_dir, 'mat_zdt_f4.csv'));
writematrix(pf1, fullfile(baseline_dir, 'mat_zdt_pf1.csv'));
writematrix(pf4, fullfile(baseline_dir, 'mat_zdt_pf4.csv'));
writematrix([hv_ex], fullfile(baseline_dir, 'mat_zdt_hv_example.csv'));
writematrix([cab], fullfile(baseline_dir, 'mat_zdt_cov_ab.csv'));
writematrix([cba], fullfile(baseline_dir, 'mat_zdt_cov_ba.csv'));
writematrix(sel, fullfile(baseline_dir, 'mat_zdt_select_front.csv'));

save(fullfile(baseline_dir, 'mat_zdt_metrics_debug.mat'), 'f1','f2','f3','f4','pf1','pf4','hv_ex','cab','cba','sel');

disp('ZDT metrics baseline exported:');
disp(baseline_dir);
