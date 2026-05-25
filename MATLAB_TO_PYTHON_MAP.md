# MATLAB 到 Python 文件对应关系

这份清单用于把旧 MATLAB 代码和当前 Python 代码对上，避免后续修改时混用旧问题定义和新 typed 50 点主线。

## 当前主线：50 点 typed-seeded

| MATLAB/旧流程概念 | Python 文件 | 备注 |
| --- | --- | --- |
| `runme.m` / 实验入口 | `run_ev_typed_mogabka_seeded.py` | 当前 MOGABKA 主入口，默认读取 50 点样本和 PCC checkpoint。 |
| 快速试跑脚本 | `run_quick.py` | 新增无空格入口，对应旧 `run me.py` 的小规模快速运行；默认不用 checkpoint，便于 GitHub clone 后冒烟检查。 |
| 多算法主比较 | `run_compare.py` + `compare/runme_50_typed_compare.py` | 统一跑 MOGABKA、HNSGA-II+KMeans、NSGA-II、NSGA-III、MOEA/D。 |
| `get_parameter.m` | `get_parameter.py` | 主线参数为 `mu=40`；旧 compare 回归仍可能使用 `mu=20`。 |
| `get_dist.m` | `get_dist.py` | 距离计算。 |
| `cal_obj.m` | `cal_obj_typed_py.py` | 新 typed 编码目标函数，决策向量长度为 `3*M`。 |
| `MOBKAGA_function.m` / `IMOBKA_funciton.m` | `mogabka_seeded.py` | Python 主算法，含外部档案、指标和 NN seed 配置。 |
| `init_sol.m` | `initialization.py` + `ev_typed_nn_seed_adapter.py` | 随机初始化与 typed 修复/打包分开。 |
| `cross.m` | `cross.py` + `compare/typed_compare_utils.py` | 基础交叉保留；typed 站点级交叉在 compare 工具中。 |
| `variate.m` / `Mutate.m` | `variate.py` + `compare/typed_compare_utils.py` | typed 站点级变异在 compare 工具中。 |
| `update_archive.m` | `update_archive.py` | 外部档案更新。 |
| `coverage.m` | `coverage.py` | 覆盖率指标。 |
| `hv_cir.m` / `hypervolume.m` | `hv.py` | HV 指标。 |
| `Spread.m` | `spread.py` | Spread 指标。 |
| `Spacing.m` | `spacing.py` | Spacing 指标。 |

## 对比算法和回归沙盒

| MATLAB 算法目录 | Python 对齐文件 | 状态 |
| --- | --- | --- |
| `NSGAII/` | `compare/nsga2_matlab_aligned.py` | 旧 MATLAB 对齐 + typed 分支。 |
| `NSGAIII/` | `compare/nsga3_matlab_aligned.py` | 旧 MATLAB 对齐 + typed 分支。 |
| `moead/` | `compare/moead_matlab_aligned.py` | 旧 MATLAB 对齐 + typed 分支。 |
| `mobka/` | `compare/mobka_stage1.py`、`compare/mobka_stage2.py`、`compare/mobka_stage3.py` | 阶段化复现/消融。 |
| `poi kmeans/` | `compare/hnsga2_kmeans_typed.py` | HNSGA-II + KMeans typed baseline。 |
| MATLAB baseline 导出 | `compare/matlab/*.m` | 生成 Python 测试用 CSV baseline。 |

## 不要混用的地方

- `compare/cal_obj_1_debug.py` 是旧单类型/旧编码目标函数，主要用于 MATLAB 数值对齐；当前主线使用 `cal_obj_typed_py.py`。
- `compare/get_parameter.py` 和根目录 `get_parameter.py` 可能代表不同阶段参数；当前 50 点主线以根目录版本为准。
- 根目录 `run_ev_typed_mogabka_seeded.py` 是当前主线；`compare/runme_matlab_aligned.py` 用于旧 MATLAB-aligned 回归。
- `charging-station-location-with-metaheuristic-chargingstation/` 是外部参考项目，不参与当前 MATLAB 到 Python 的主迁移链路，也不纳入本次 GitHub 上传范围。

## 推荐维护顺序

1. 先改 `cal_obj_typed_py.py`、`ev_typed_nn_seed_adapter.py`、`get_parameter.py` 等问题定义层。
2. 再改 `mogabka_seeded.py` 或 `compare/*_matlab_aligned.py` 的算法层。
3. 最后改 `run_ev_typed_mogabka_seeded.py`、`run_compare.py` 等实验入口。
4. 修改后优先跑 `python run_quick.py`，再进入 `compare` 目录跑 `pytest`。
