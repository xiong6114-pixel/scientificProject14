import os
import json
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import h5py
import numpy as np
import torch

from cal_obj_typed_py import cal_obj_typed, build_default_type_info, build_distance_matrix
from ev_typed_nn_seed_adapter import build_ev_typed_problem_context
from get_mofcn import set_ev_context, clear_ev_context
from get_parameter import get_parameter
from mogabka_seeded import MOGABKA
from nn_seed_injection_50 import make_pcc_seed_builder, SeedDecodeConfig


PROJECT_DIR = Path(__file__).resolve().parent
POI_KMEANS_DIR = PROJECT_DIR.parent / "poi kmeans"

DEFAULT_DEBUG_MAT_PATH = POI_KMEANS_DIR / "dataset_v3_50_debug.mat"
DEFAULT_DEMAND_CSV = PROJECT_DIR / "demand_points_info.csv"
DEFAULT_CHARGE_CSV = PROJECT_DIR / "charge_points_info.csv"
DEFAULT_X_ONE_CELL = PROJECT_DIR / "x_one_cell_flat.csv"
DEFAULT_CKPT_PATH = PROJECT_DIR / "set_transformer_pcc_ckpt_50.pt"


def _load_csv_2d(path: Path) -> np.ndarray:
    arr = np.loadtxt(path, delimiter=",", dtype=float)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    return np.asarray(arr, dtype=float)


def _read_h5_cell_2d(f: h5py.File, ds: h5py.Dataset, sample_idx: int) -> np.ndarray:
    ref = ds[0, sample_idx] if ds.shape[0] == 1 else ds[sample_idx, 0]
    arr = np.array(f[ref], dtype=float)
    if arr.ndim != 2:
        raise ValueError(f"Unexpected cell payload ndim={arr.ndim}")
    if arr.shape[0] == 3 and arr.shape[1] != 3:
        arr = arr.T
    return np.asarray(arr, dtype=float)


def _load_runtime_from_debug_mat(mat_path: Path, sample_idx: int):
    with h5py.File(mat_path, "r") as f:
        x_all = f["X_all"]
        if x_all.ndim != 2:
            raise ValueError(f"X_all ndim should be 2, got {x_all.ndim}")

        if x_all.shape[0] == 2450:
            n_samples = x_all.shape[1]
            if not (0 <= sample_idx < n_samples):
                raise IndexError(f"sample_idx={sample_idx} out of range [0, {n_samples - 1}]")
            x_raw_flat = np.array(x_all[:, sample_idx], dtype=np.float32).reshape(-1)
        elif x_all.shape[1] == 2450:
            n_samples = x_all.shape[0]
            if not (0 <= sample_idx < n_samples):
                raise IndexError(f"sample_idx={sample_idx} out of range [0, {n_samples - 1}]")
            x_raw_flat = np.array(x_all[sample_idx, :], dtype=np.float32).reshape(-1)
        else:
            raise ValueError(f"X_all shape does not look like 50-point flat features: {x_all.shape}")

        demand_points_info = _read_h5_cell_2d(f, f["D_samples"], sample_idx)
        charge_points_info = _read_h5_cell_2d(f, f["C_samples"], sample_idx)

    return (
        np.asarray(demand_points_info, dtype=float),
        np.asarray(charge_points_info, dtype=float),
        np.asarray(x_raw_flat, dtype=np.float32),
        f"debug_mat:{mat_path.name}[sample={sample_idx}]",
    )


def _load_runtime_from_csv(demand_path: Path, charge_path: Path, x_path: Path):
    demand_points_info = _load_csv_2d(demand_path)
    charge_points_info = _load_csv_2d(charge_path)
    x_raw_flat = np.loadtxt(x_path, delimiter=",", dtype=np.float32).reshape(-1)
    return (
        np.asarray(demand_points_info, dtype=float),
        np.asarray(charge_points_info, dtype=float),
        np.asarray(x_raw_flat, dtype=np.float32),
        f"csv:{demand_path.name},{charge_path.name},{x_path.name}",
    )


def _load_runtime_inputs(expected_points: int, expected_in_dim: int):
    debug_mat_path = Path(os.environ.get("RUNTIME_DEBUG_MAT_PATH", str(DEFAULT_DEBUG_MAT_PATH)))
    sample_idx = int(os.environ.get("RUNTIME_SAMPLE_INDEX", "0"))

    if debug_mat_path.is_file():
        demand_points_info, charge_points_info, x_raw_flat, source_desc = _load_runtime_from_debug_mat(
            debug_mat_path,
            sample_idx,
        )
    else:
        demand_path = Path(os.environ.get("DEMAND_POINTS_PATH", str(DEFAULT_DEMAND_CSV)))
        charge_path = Path(os.environ.get("CHARGE_POINTS_PATH", str(DEFAULT_CHARGE_CSV)))
        x_path = Path(os.environ.get("X_ONE_CELL_PATH", str(DEFAULT_X_ONE_CELL)))
        demand_points_info, charge_points_info, x_raw_flat, source_desc = _load_runtime_from_csv(
            demand_path,
            charge_path,
            x_path,
        )

    if demand_points_info.ndim != 2 or demand_points_info.shape[1] != 3:
        raise ValueError(f"demand_points_info should be [N,3], got {demand_points_info.shape}")
    if charge_points_info.ndim != 2 or charge_points_info.shape[1] != 3:
        raise ValueError(f"charge_points_info should be [M,3], got {charge_points_info.shape}")

    if charge_points_info.shape[0] != expected_points:
        raise ValueError(
            f"Charge point count {charge_points_info.shape[0]} does not match checkpoint n_points={expected_points}"
        )
    if x_raw_flat.size != expected_in_dim:
        raise ValueError(
            f"x_raw_flat length {x_raw_flat.size} does not match checkpoint in_dim={expected_in_dim}"
        )

    return demand_points_info, charge_points_info, x_raw_flat, source_desc


def _summarize_solution(sol: np.ndarray) -> dict:
    sol = np.round(np.asarray(sol, dtype=float).reshape(-1)).astype(int)
    m = sol.size // 3
    n_slow = sol[:m]
    n_fast = sol[m : 2 * m]
    n_ultra = sol[2 * m :]
    total = n_slow + n_fast + n_ultra
    built_idx = np.where(total > 0)[0]
    order = built_idx[np.argsort(-total[built_idx])] if built_idx.size > 0 else np.empty((0,), dtype=int)

    return {
        "built_station_count": int(built_idx.size),
        "total_piles": int(total.sum()),
        "slow_piles": int(n_slow.sum()),
        "fast_piles": int(n_fast.sum()),
        "ultra_piles": int(n_ultra.sum()),
        "top_station_totals": [
            {"station_index": int(j), "total_piles": int(total[j])}
            for j in order[:10]
        ],
    }


def _select_representative_indices(fit: np.ndarray) -> dict:
    fit = np.asarray(fit, dtype=float)
    mins = fit.min(axis=0)
    maxs = fit.max(axis=0)
    fit_norm = (fit - mins) / np.maximum(maxs - mins, 1e-12)
    knee_idx = int(np.argmin(np.sqrt(np.sum(fit_norm ** 2, axis=1))))
    return {
        "min_obj1": int(np.argmin(fit[:, 0])),
        "min_obj2": int(np.argmin(fit[:, 1])),
        "knee": knee_idx,
    }


def main():
    torch_threads = int(os.environ.get("TORCH_NUM_THREADS", "1"))
    try:
        torch.set_num_threads(torch_threads)
        torch.set_num_interop_threads(1)
    except Exception:
        pass

    ckpt_path = Path(os.environ.get("CKPT_PATH", str(DEFAULT_CKPT_PATH)))
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    seed_device = os.environ.get("SEED_DEVICE", "cpu")
    max_iter = int(os.environ.get("MAX_ITER", "50"))
    search_agents_no = int(os.environ.get("SEARCH_AGENTS_NO", "50"))
    init_nn_seed_count = int(os.environ.get("INIT_NN_SEED_COUNT", "12"))

    print("loading checkpoint...", flush=True)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    expected_points = int(ckpt["n_points"])
    feat_dim = int(ckpt["feat_dim"])
    expected_in_dim = int(ckpt.get("in_dim", expected_points * feat_dim))

    print("loading runtime sample...", flush=True)
    demand_points_info, charge_points_info, x_raw_flat, source_desc = _load_runtime_inputs(
        expected_points=expected_points,
        expected_in_dim=expected_in_dim,
    )

    print("building typed context...", flush=True)
    parameter = get_parameter()
    typeInfo = build_default_type_info(
        parameter=parameter,
        nmax=np.array([20, 20, 10], dtype=int),
    )

    M = charge_points_info.shape[0]
    dist_matrix = build_distance_matrix(
        demand_points_info[:, 0:2],
        charge_points_info[:, 0:2],
    )

    problem_context = build_ev_typed_problem_context(
        x_raw_flat=x_raw_flat,
        charge_points_num=M,
        cal_obj_fn=cal_obj_typed,
        demand_points_info=demand_points_info,
        charge_points_info=charge_points_info,
        parameter=parameter,
        typeInfo=typeInfo,
        type_prob=(0.50, 0.35, 0.15),
        fallback_index=0,
        stochastic_split=False,
        dist_matrix=dist_matrix,
    )

    set_ev_context(problem_context)

    decode_cfg = SeedDecodeConfig(
        num_seeds=init_nn_seed_count,
        count_radius=2,
        top_margin=4,
        stochastic_ratio=0.75,
        priority_temperature=1.0,
        capacity_temperature=1.0,
        min_k=1,
        max_k=M,
    )

    seed_builder = make_pcc_seed_builder(
        ckpt_path=str(ckpt_path),
        decode_cfg=decode_cfg,
        device=seed_device,
    )

    seed_cfg = {
        "enabled": True,
        "init_enabled": True,
        "init_nn_seed_count": init_nn_seed_count,
        "builder_fn": seed_builder,
        "reinject_enabled": False,
        "reinject_generations": (20, 60),
        "reinject_count": 4,
    }

    ub_vec = np.concatenate(
        [
            np.full(M, typeInfo.nmax[0], dtype=float),
            np.full(M, typeInfo.nmax[1], dtype=float),
            np.full(M, typeInfo.nmax[2], dtype=float),
        ]
    )

    print(f"runtime source: {source_desc}")
    print(f"checkpoint: {ckpt_path.name}")
    print(f"n_points={expected_points}, feat_dim={feat_dim}, x_raw_dim={x_raw_flat.size}")
    print(f"demand_points={demand_points_info.shape[0]}, charge_points={charge_points_info.shape[0]}")
    print(f"level_to_piles={ckpt.get('level_to_piles', None)}")
    print(f"parameter={parameter.tolist()}")
    print(f"seed_device={seed_device}, torch_threads={torch_threads}")
    print(f"max_iter={max_iter}, search_agents_no={search_agents_no}, init_nn_seed_count={init_nn_seed_count}")
    print("starting MOGABKA...", flush=True)

    try:
        Fitness, POP, turePF, Result = MOGABKA(
            Max_iter=max_iter,
            SearchAgents_no=search_agents_no,
            FUN="EV_TYPED_CS",
            dim=3 * M,
            numObj=2,
            lb=0,
            ub=ub_vec,
            seed=42,
            seed_injection_config=seed_cfg,
            problem_context=problem_context,
        )
    finally:
        clear_ev_context()

    print("Finished.")
    print("Fitness shape:", Fitness.shape)
    print("POP shape:", POP.shape)
    print("turePF shape:", turePF.shape)

    archive = np.asarray(Result.get("archive", np.empty((0, 3 * M))), dtype=float)
    archive_fitness = np.asarray(Result.get("archive_fitness", np.empty((0, 2))), dtype=float)
    archive_pf_pop = np.asarray(Result.get("archive_pf_pop", np.empty((0, 3 * M))), dtype=float)
    archive_pf_fitness = np.asarray(Result.get("archive_pf_fitness", np.empty((0, 2))), dtype=float)

    print("\nLast generation metrics:")
    for k, v in Result.items():
        if isinstance(v, np.ndarray) and v.ndim == 1 and v.size > 0:
            print(k, v[-1])

    rep_summary = {}
    if archive_pf_fitness.size > 0 and archive_pf_pop.size > 0:
        rep_idx = _select_representative_indices(archive_pf_fitness)
        for name, idx in rep_idx.items():
            rep_summary[name] = {
                "fitness": archive_pf_fitness[idx].tolist(),
                **_summarize_solution(archive_pf_pop[idx]),
            }

        print("\nRepresentative solutions:")
        for name in ("min_obj1", "knee", "min_obj2"):
            item = rep_summary[name]
            print(
                f"{name}: fitness={item['fitness']} built={item['built_station_count']} "
                f"total_piles={item['total_piles']} slow/fast/ultra="
                f"{[item['slow_piles'], item['fast_piles'], item['ultra_piles']]}"
            )

    np.savetxt(PROJECT_DIR / "final_fitness_typed.csv", Fitness, delimiter=",")
    np.savetxt(PROJECT_DIR / "final_pop_typed.csv", POP, delimiter=",")
    np.savetxt(PROJECT_DIR / "final_pf_typed.csv", turePF, delimiter=",")
    np.savetxt(PROJECT_DIR / "archive_pop_typed.csv", archive, delimiter=",")
    np.savetxt(PROJECT_DIR / "archive_fitness_typed.csv", archive_fitness, delimiter=",")
    np.savetxt(PROJECT_DIR / "archive_pf_pop_typed.csv", archive_pf_pop, delimiter=",")
    np.savetxt(PROJECT_DIR / "archive_pf_fitness_typed.csv", archive_pf_fitness, delimiter=",")

    rep_payload = {
        "runtime_source": source_desc,
        "checkpoint": ckpt_path.name,
        "parameter": parameter.tolist(),
        "level_to_piles": ckpt.get("level_to_piles", None),
        "representatives": rep_summary,
    }
    with open(PROJECT_DIR / "representative_solutions_typed.json", "w", encoding="utf-8") as f:
        json.dump(rep_payload, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
