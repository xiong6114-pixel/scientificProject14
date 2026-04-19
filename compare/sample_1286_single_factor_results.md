# sample_idx=1286 单因素结果

基线：当前默认链路，`max_iter=50`，`popnum=50`，`mu=40`，`V=40`，`cross=0.8`，`mutate=0.3`，`archive=100`，动态参考前沿 HV。

## nnseed 结果对比

| 设置 | nnseed HV | nnseed ND 点数 | nnseed -> MOBKA | nnseed -> NSGA2 | nnseed -> NSGA3 | nnseed -> MOEAD | NSGA2 -> nnseed | NSGA3 -> nnseed | MOEAD -> nnseed |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 默认基线 | 0.7245071950 | 未单独存档 | 1.000000 | 0.711538 | 0.966667 | 1.000000 | 0.250000 | 0.000000 | 0.000000 |
| 只改 `mu=10, V=35` | 0.5846983809 | 96 | 1.000000 | 0.000000 | 0.000000 | 0.000000 | 0.468750 | 0.447917 | 0.416667 |
| 只改 `cross=0.3, mutate=0.8` | 0.8025199078 | 96 | 1.000000 | 0.983607 | 0.925926 | 0.903226 | 0.000000 | 0.031250 | 0.031250 |

## 各算法 HV

### 默认基线

- `IMOGABKA-seed`: `0.7245071950`
- `NSGA2`: `0.6403134522`
- `NSGA3`: `0.6169763550`
- `MOEAD`: `0.4825370999`
- `MOBKA`: `0.3275589292`

### 只改 `mu=10, V=35`

- `NSGA2`: `0.6425732267`
- `NSGA3`: `0.6247773475`
- `MOEAD`: `0.5973725274`
- `IMOGABKA-seed`: `0.5846983809`
- `MOBKA`: `0.4109513698`

### 只改 `cross=0.3, mutate=0.8`

- `IMOGABKA-seed`: `0.8025199078`
- `NSGA2`: `0.7302523206`
- `NSGA3`: `0.7289280305`
- `MOEAD`: `0.6776873529`
- `MOBKA`: `0.2718978663`

## 结果文件

- 只改 `mu,V`
  - `scientificProject14/compare/real_batch_samples_1286_only_mu_v_summary_by_algorithm.csv`
  - `scientificProject14/compare/real_batch_samples_1286_only_mu_v_coverage_summary.csv`
- 只改 `cross/mutate`
  - `scientificProject14/compare/real_batch_samples_1286_only_cross_mutate_summary_by_algorithm.csv`
  - `scientificProject14/compare/real_batch_samples_1286_only_cross_mutate_coverage_summary.csv`
