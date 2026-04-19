# sample_idx=1286 重跑结果（按论文参数口径）

## 本次使用设置

- `sample_idx = 1286`
- `max_iter = 200`
- `popnum = 50`
- 固定 `HV` 参考点：`(200, 500000000)`
- 模型参数：`Te=0, mu=10, V=35, Cj=12000, Co=5`
- 统一交叉/变异概率：`cross_rate=0.3`, `variate_rate=0.8`
- `nnseed(IMOGABKA-seed)`：`archive_size=300`, `Levy beta=1.5`
- `MOEAD` 邻居大小请求为 `100`，但由于 `popnum=50`，实际生效上限为 `50`

## HV

| 算法 | HV |
| --- | ---: |
| IMOGABKA-seed | 95690246656.00595 |
| NSGA2 | 94414354361.90782 |
| NSGA3 | 91141274182.06871 |
| MOEAD | 89721358923.90474 |
| MOBKA | 86035993524.16304 |

## Coverage

说明：`A -> B` 表示 “A 覆盖 B 的比例”。

| 方向 | Coverage |
| --- | ---: |
| IMOGABKA-seed -> MOBKA | 1.000000 |
| MOBKA -> IMOGABKA-seed | 0.000000 |
| IMOGABKA-seed -> NSGA2 | 0.080645 |
| NSGA2 -> IMOGABKA-seed | 0.918819 |
| IMOGABKA-seed -> NSGA3 | 0.000000 |
| NSGA3 -> IMOGABKA-seed | 0.911439 |
| IMOGABKA-seed -> MOEAD | 0.000000 |
| MOEAD -> IMOGABKA-seed | 0.892989 |

## 结果文件

- 详细结果：`scientificProject14/compare/real_batch_samples_1286_1286_detailed.csv`
- HV 汇总：`scientificProject14/compare/real_batch_samples_1286_1286_summary_by_algorithm.csv`
- Coverage 汇总：`scientificProject14/compare/real_batch_samples_1286_1286_coverage_summary.csv`
