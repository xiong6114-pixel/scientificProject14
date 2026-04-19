# sample_idx 1285-1300 nnseed 对比汇总

说明：Coverage(A->B) 表示算法 A 的前沿支配算法 B 前沿点集的比例。

## HV 总览

- nnseed(IMOGABKA-seed) 在 16/16 个样本上 HV 均为第一。
- nnseed 平均 HV = 0.872636，中位数 HV = 0.889364。

| 算法 | runs | hv_mean | hv_median | hv_std |
|---|---:|---:|---:|---:|
| IMOGABKA-seed | 16 | 0.872636 | 0.889364 | 0.060668 |
| NSGA2 | 16 | 0.808862 | 0.818279 | 0.068216 |
| NSGA3 | 16 | 0.802808 | 0.812506 | 0.069798 |
| MOEAD | 16 | 0.719578 | 0.740863 | 0.079987 |
| MOBKA | 16 | 0.417501 | 0.422516 | 0.062489 |

## nnseed Coverage 均值

| Coverage 方向 | mean | median | std |
|---|---:|---:|---:|
| nnseed -> MOBKA | 1.000000 | 1.000000 | 0.000000 |
| MOBKA -> nnseed | 0.000000 | 0.000000 | 0.000000 |
| nnseed -> NSGA2 | 0.718581 | 0.796552 | 0.208049 |
| NSGA2 -> nnseed | 0.096529 | 0.051815 | 0.094378 |
| nnseed -> NSGA3 | 0.769343 | 0.893920 | 0.266520 |
| NSGA3 -> nnseed | 0.041910 | 0.015793 | 0.054074 |
| nnseed -> MOEAD | 0.916471 | 1.000000 | 0.197779 |
| MOEAD -> nnseed | 0.008065 | 0.000000 | 0.021337 |

## 每个样本的 HV 与双向 Coverage

详表见 CSV：.\real_batch_samples_1285_1300_nnseed_vs_baselines_by_sample.csv
