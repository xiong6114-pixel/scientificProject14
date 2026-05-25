# MATLAB 到 Python 对齐项目

本目录是充电站选址多目标优化代码的 Python 整理版，核心目标是把原 MATLAB 算法链路迁移到可复现实验的 Python 工程中。当前主线已经切到 **50 个候选点 + 三类型充电桩编码 + 神经网络种子注入 + MOGABKA/对比算法评估**。

## 目录主线

| 路径 | 作用 |
| --- | --- |
| `run_quick.py` | 快速冒烟运行入口，默认小迭代数和随机初始化；即使没有本地 checkpoint 也能确认环境和数据是否跑通。 |
| `run_ev_typed_mogabka_seeded.py` | 当前主实验入口：加载 50 点样本、模型 checkpoint，运行带 NN seed 的 MOGABKA。 |
| `mogabka_seeded.py` | Python 版 MOGABKA 主体，含 typed 编码、种子注入、外部档案和指标输出。 |
| `cal_obj_typed_py.py` | typed 充电站目标函数，对应 MATLAB `cal_obj` 的新编码版本。 |
| `ev_typed_nn_seed_adapter.py` | 神经网络输出到 typed 决策向量的转换、修复和上下文构建。 |
| `nn_seed_injection_50.py` | 50 点 PCC 模型的种子生成器。 |
| `compare/` | MATLAB 对齐回归、NSGA-II/NSGA-III/MOEA/D/HNSGA-II+KMeans 对比实验。 |
| `compare/matlab/` | 生成 MATLAB baseline 的导出脚本，用于 Python 回归测试。 |

更细的 MATLAB/Python 文件对应关系见 `MATLAB_TO_PYTHON_MAP.md`。

## 环境

建议使用独立虚拟环境：

```powershell
cd path\to\scientificProject14
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

如果只跑核心 typed 主线，主要依赖是 `numpy`、`torch`、`h5py`、`scipy`、`matplotlib`、`scikit-learn`。外部/旧版参考代码不属于当前主线，已从上传范围中排除。

## 常用运行方式

快速检查：

```powershell
python run_quick.py
```

带神经网络种子注入的 MOGABKA typed-seeded 运行需要本地 checkpoint。默认路径是 `set_transformer_pcc_ckpt_50.pt`，也可以用 `CKPT_PATH` 指向其他模型文件：

```powershell
$env:MAX_ITER = "50"
$env:SEARCH_AGENTS_NO = "50"
$env:INIT_NN_SEED_COUNT = "12"
$env:SEED_DEVICE = "cpu"
python run_ev_typed_mogabka_seeded.py
```

如果只想不用 checkpoint 先跑通算法链路：

```powershell
$env:INIT_NN_SEED_COUNT = "0"
python run_ev_typed_mogabka_seeded.py
```

多算法对比：

```powershell
python run_compare.py
```

对比初始化模式可用 `COMPARE_INIT_MODE` 控制：

| 模式 | 含义 |
| --- | --- |
| `all_random` | 所有算法使用同一份随机 typed 初始种群。 |
| `all_shared_seeded_init` | 所有算法共享包含 NN seed 的 typed 初始种群。 |
| `mobka_only_seeded` | 只有 MOGABKA 使用 NN seed，其他算法使用共享随机初始种群。 |
| `all_modes` | 依次跑完上面三种模式并汇总。 |

示例：

```powershell
$env:COMPARE_INIT_MODE = "all_modes"
$env:MAX_ITER = "30"
python run_compare.py
```

## 数据和输出

默认输入：

- `demand_points_info.csv`
- `charge_points_info.csv`
- `x_one_cell_flat.csv`
- `set_transformer_pcc_ckpt_50.pt`（仅 NN seed 实验需要；模型文件默认不上传 GitHub）

如果存在 `..\poi kmeans\dataset_v3_50_debug.mat`，运行时会优先读取其中的样本；否则读取上面的 CSV。可通过环境变量覆盖：

- `RUNTIME_DEBUG_MAT_PATH`
- `RUNTIME_SAMPLE_INDEX`
- `DEMAND_POINTS_PATH`
- `CHARGE_POINTS_PATH`
- `X_ONE_CELL_PATH`
- `CKPT_PATH`

主线输出包括以下可再生成文件，默认不会上传 GitHub：

- `final_*_typed.csv`
- `archive_*_typed.csv`
- `representative_solutions_typed.json`

对比实验输出保存在 `compare/`，文件名前缀通常为 `py_runme50_typed_`、`real_batch_` 或 `nnseed_`。

## 测试

MATLAB 对齐测试在 `compare/tests/`：

```powershell
cd compare
pytest
```

部分测试需要先由 MATLAB 脚本导出 baseline 文件；缺少 baseline 时测试会自动跳过。
