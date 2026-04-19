import os

import numpy as np

from coverage import Coverage
from cross import cross
from environmental_selection import EnvironmentalSelection
from gd import GD
from get_mofcn import getMOFcn
from hv import HV
from igd import IGD
from initialization import initialization
from levy import levy
from spacing import Spacing
from spread import Spread
from uniform_point import UniformPoint
from update_archive import update_archive
from variate import variate


DEFAULT_SEED_INJECTION_CONFIG = {
    "enabled": False,
    "init_enabled": True,
    "init_nn_seed_count": 12,
    "initial_individuals": None,      # 可直接传 [K,dim]
    "builder_fn": None,               # builder_fn(num_needed, dim, lb_vec, ub_vec, context)->[K,dim]
    "reinject_enabled": False,
    "reinject_generations": (20, 60),
    "reinject_count": 4,
}


def MOGABKA(
    Max_iter,
    SearchAgents_no,
    FUN,
    dim,
    numObj,
    lb,
    ub,
    seed=None,
    seed_injection_config=None,
    problem_context=None,
):
    if seed is not None:
        np.random.seed(int(seed))

    max_iter = int(Max_iter)
    search_agents_no = int(SearchAgents_no)
    dim = int(dim)
    num_obj = int(numObj)

    lb_vec = _expand_bound(lb, dim)
    ub_vec = _expand_bound(ub, dim)

    cfg = DEFAULT_SEED_INJECTION_CONFIG.copy()
    if seed_injection_config is not None:
        cfg.update(seed_injection_config)

    problem_context = problem_context or {}
    integer_encoding = bool(problem_context.get("integer_encoding", False))
    invalid_penalty = float(problem_context.get("invalid_penalty", 1e12))
    debug_seed = bool(problem_context.get("debug_seed_injection", False))
    debug_metrics = bool(problem_context.get("debug_metrics", False))

    Iter = 1
    variate_rate = float(problem_context.get("variate_rate", os.environ.get("VARIATE_RATE", "0.3")))
    cross_rate = float(problem_context.get("cross_rate", os.environ.get("CROSS_RATE", "0.8")))
    Plevy = float(problem_context.get("p_levy", os.environ.get("P_LEVY", "0.5")))
    archive = np.empty((0, dim), dtype=float)
    archive_size = int(problem_context.get("archive_size", os.environ.get("ARCHIVE_SIZE", "100")))

    Result = {
        "IGD": np.zeros(max_iter, dtype=float),
        "GD": np.zeros(max_iter, dtype=float),
        "HV": np.zeros(max_iter, dtype=float),
        "Spacing": np.zeros(max_iter, dtype=float),
        "Spread": np.zeros(max_iter, dtype=float),
        "Coverage": np.zeros(max_iter, dtype=float),
    }

    Fitness = np.empty((0, num_obj), dtype=float)
    POP = np.empty((0, dim), dtype=float)
    turePF = np.empty((0, num_obj), dtype=float)

    while Iter <= max_iter:
        if Iter == 1:
            POP_rand = initialization(search_agents_no, dim, ub_vec, lb_vec)
            POP_rand = _sanitize_population(
                POP_rand,
                dim,
                lb_vec,
                ub_vec,
                integer_encoding=integer_encoding,
            )

            if cfg["enabled"] and cfg["init_enabled"]:
                POP_seed = _build_seed_population(
                    cfg=cfg,
                    count=int(cfg["init_nn_seed_count"]),
                    dim=dim,
                    lb_vec=lb_vec,
                    ub_vec=ub_vec,
                    context=problem_context,
                    integer_encoding=integer_encoding,
                )
                if debug_seed:
                    print(f"[init] random={POP_rand.shape[0]} seed={POP_seed.shape[0]} target={search_agents_no}")
                POP = _merge_seed_and_random(
                    seed_pop=POP_seed,
                    random_pop=POP_rand,
                    target_size=search_agents_no,
                    dim=dim,
                    lb_vec=lb_vec,
                    ub_vec=ub_vec,
                    integer_encoding=integer_encoding,
                )
            else:
                POP = _fill_population_to_size(
                    POP_rand,
                    target_size=search_agents_no,
                    dim=dim,
                    lb_vec=lb_vec,
                    ub_vec=ub_vec,
                    integer_encoding=integer_encoding,
                )

            Z, search_agents_no_new = UniformPoint(search_agents_no, num_obj)
            search_agents_no = int(np.asarray(search_agents_no_new).item())

            POP = _fill_population_to_size(
                POP,
                target_size=search_agents_no,
                dim=dim,
                lb_vec=lb_vec,
                ub_vec=ub_vec,
                integer_encoding=integer_encoding,
            )

            Fitness, _, _ = getMOFcn(FUN, POP, num_obj)
            Zmin = np.min(Fitness, axis=0)
            POP = EnvironmentalSelection(FUN, POP, search_agents_no, num_obj, Z, Zmin)
            POP = _sanitize_population(
                POP,
                dim,
                lb_vec,
                ub_vec,
                integer_encoding=integer_encoding,
            )
            POP_old = POP.copy()

        bip = np.random.randint(1, search_agents_no + 1)
        XLeader_Pos = POP_old[bip - 1, :].copy()

        XPos = POP
        XPosNew1 = np.zeros((search_agents_no, dim), dtype=float)
        XPosNew2 = np.zeros((search_agents_no, dim), dtype=float)
        XPos_new3_list = []
        XPos_new4_list = []
        XPos_new5_list = []

        for i in range(search_agents_no):
            if np.random.rand() < variate_rate:
                new_sol = variate(XPos[i, :], dim, 1)
                XPos_new3_list.append(new_sol)

            if np.random.rand() < cross_rate:
                rand_id = np.random.randint(1, search_agents_no + 1)
                new_sol = cross(XPos[i, :], dim, XPos[rand_id - 1, :])
                XPos_new4_list.append(new_sol)

            p = 0.9
            r = np.random.rand()
            n = 0.05 * np.exp(-2 * (Iter / max_iter) ** 2)
            if p < r:
                XPosNew = XPos[i, :] + n * (1 + np.sin(r)) * XPos[i, :]
            else:
                XPosNew = XPos[i, :] * (n * (2 * np.random.rand(dim) - 1) + 1)

            Flag4ub = XPosNew >= ub_vec
            Flag4lb = XPosNew <= lb_vec
            XPosNew1[i, :] = (XPosNew * (~(Flag4ub | Flag4lb))) + ub_vec * Flag4ub + lb_vec * Flag4lb

            m = 2 * np.sin(r + np.pi / 2)
            ori_value = np.random.rand(dim)
            cauchy_value = np.tan((ori_value - 0.5) * np.pi)
            if np.random.rand() < np.random.rand():
                XPosNew = XPos[i, :] + cauchy_value[dim - 1] * (XPos[i, :] - XLeader_Pos)
            else:
                XPosNew = XPos[i, :] + cauchy_value[dim - 1] * (XLeader_Pos - m * XPos[i, :])

            Flag4ub = XPosNew >= ub_vec
            Flag4lb = XPosNew <= lb_vec
            XPosNew2[i, :] = (XPosNew * (~(Flag4ub | Flag4lb))) + ub_vec * Flag4ub + lb_vec * Flag4lb

        XPos_new3 = np.vstack(XPos_new3_list) if len(XPos_new3_list) > 0 else np.empty((0, dim), dtype=float)
        XPos_new4 = np.vstack(XPos_new4_list) if len(XPos_new4_list) > 0 else np.empty((0, dim), dtype=float)

        POP = np.vstack([POP_old, XPosNew1, XPosNew2, XPos_new4, XPos_new3])
        POP = _sanitize_population(
            POP,
            dim,
            lb_vec,
            ub_vec,
            integer_encoding=integer_encoding,
        )

        for i in range(POP.shape[0]):
            if np.random.rand() < Plevy:
                L = levy(dim)

                if integer_encoding:
                    new_sol = np.round(POP[i, :] + POP[i, :] * L)
                    new_sol = np.clip(new_sol, lb_vec, ub_vec)
                else:
                    # 保留 benchmark 连续编码下的原逻辑
                    new_sol = np.round(POP[i, :] + POP[i, :] * L) / 25.0
                    new_sol[new_sol > 1] = 1
                    new_sol[new_sol < 0] = 0

                XPos_new5_list.append(new_sol)

        XPos_new5 = np.vstack(XPos_new5_list) if len(XPos_new5_list) > 0 else np.empty((0, dim), dtype=float)
        POP = np.vstack([POP, XPos_new5])
        POP = _sanitize_population(
            POP,
            dim,
            lb_vec,
            ub_vec,
            integer_encoding=integer_encoding,
        )

        subfit, _, turePF = getMOFcn(FUN, POP, num_obj)
        Zmin = np.min(np.vstack([Zmin, subfit]), axis=0)
        POP = EnvironmentalSelection(FUN, POP, search_agents_no, num_obj, Z, Zmin)
        POP = _sanitize_population(
            POP,
            dim,
            lb_vec,
            ub_vec,
            integer_encoding=integer_encoding,
        )

        # ===== 可选：中途重注入 =====
        if cfg["enabled"] and cfg["reinject_enabled"] and (Iter in set(cfg["reinject_generations"])):
            reinjected = _build_seed_population(
                cfg=cfg,
                count=int(cfg["reinject_count"]),
                dim=dim,
                lb_vec=lb_vec,
                ub_vec=ub_vec,
                context=problem_context,
                integer_encoding=integer_encoding,
            )
            if debug_seed:
                print(f"[reinject] gen={Iter} count={reinjected.shape[0]}")
            if reinjected.shape[0] > 0:
                POP = np.vstack([POP, reinjected])
                POP = _sanitize_population(
                    POP,
                    dim,
                    lb_vec,
                    ub_vec,
                    integer_encoding=integer_encoding,
                )
                POP = EnvironmentalSelection(FUN, POP, search_agents_no, num_obj, Z, Zmin)
                POP = _sanitize_population(
                    POP,
                    dim,
                    lb_vec,
                    ub_vec,
                    integer_encoding=integer_encoding,
                )

        POP_old = POP.copy()
        archive = update_archive(archive, POP, archive_size, FUN, num_obj)
        archive = _sanitize_population(
            archive,
            dim,
            lb_vec,
            ub_vec,
            integer_encoding=integer_encoding,
        )

        Fitness, _, archive_pf = getMOFcn(FUN, archive, num_obj)
        if turePF.shape[0] == 0 and archive_pf.shape[0] > 0:
            turePF = archive_pf

        valid_fit = _filter_metric_points(Fitness, invalid_penalty)
        valid_pf = _filter_metric_points(turePF, invalid_penalty)
        if valid_pf.shape[0] == 0:
            valid_pf = _filter_metric_points(archive_pf, invalid_penalty)

        Result["IGD"][Iter - 1] = _safe_igd(valid_fit, valid_pf)
        Result["GD"][Iter - 1] = _safe_gd(valid_fit, valid_pf)
        Result["Spacing"][Iter - 1] = _safe_spacing(valid_fit, valid_pf)
        Result["Spread"][Iter - 1] = _safe_spread(valid_fit, valid_pf)
        Result["HV"][Iter - 1] = _safe_hv(valid_fit, valid_pf)
        Result["Coverage"][Iter - 1] = _safe_coverage(valid_fit, valid_pf)

        if debug_metrics:
            print(
                f"[iter {Iter}] archive={archive.shape[0]} feasible_archive={valid_fit.shape[0]} "
                f"pf={valid_pf.shape[0]} hv={Result['HV'][Iter - 1]:.6g}"
            )
        else:
            print(Iter)
        Iter += 1

    archive_fit = Fitness.copy()
    archive_pf_pop, archive_pf_fit = _extract_feasible_first_front_archive(
        archive=archive,
        archive_fit=archive_fit,
        invalid_penalty=invalid_penalty,
    )
    Result["archive"] = archive.copy()
    Result["archive_fitness"] = archive_fit
    Result["archive_pf_pop"] = archive_pf_pop
    Result["archive_pf_fitness"] = archive_pf_fit

    return Fitness, POP, turePF, Result


def _build_seed_population(cfg, count, dim, lb_vec, ub_vec, context, integer_encoding=False):
    parts = []

    initial_individuals = cfg.get("initial_individuals", None)
    if initial_individuals is not None:
        arr = _sanitize_population(
            initial_individuals,
            dim,
            lb_vec,
            ub_vec,
            integer_encoding=integer_encoding,
        )
        if arr.shape[0] > 0:
            parts.append(arr)

    builder_fn = cfg.get("builder_fn", None)
    if builder_fn is not None and count > 0:
        arr = builder_fn(count, dim, lb_vec, ub_vec, context)
        arr = _sanitize_population(
            arr,
            dim,
            lb_vec,
            ub_vec,
            integer_encoding=integer_encoding,
        )
        if arr.shape[0] > 0:
            parts.append(arr)

    if len(parts) == 0:
        return np.empty((0, dim), dtype=float)

    out = np.vstack(parts)
    out = np.unique(out, axis=0)
    return out


def _sanitize_population(pop, dim, lb_vec, ub_vec, integer_encoding=False):
    pop = np.asarray(pop, dtype=float)
    if pop.size == 0:
        return np.empty((0, dim), dtype=float)

    if pop.ndim == 1:
        pop = pop.reshape(1, -1)

    if pop.shape[1] != dim:
        raise ValueError(f"种群维度应为 {dim}，实际 {pop.shape[1]}")

    pop = np.clip(pop, lb_vec, ub_vec)

    if integer_encoding:
        pop = np.round(pop)

    pop = np.unique(pop, axis=0)
    return pop


def _fill_population_to_size(pop, target_size, dim, lb_vec, ub_vec, integer_encoding=False):
    pop = _sanitize_population(
        pop,
        dim,
        lb_vec,
        ub_vec,
        integer_encoding=integer_encoding,
    )

    while pop.shape[0] < target_size:
        extra = initialization(target_size - pop.shape[0], dim, ub_vec, lb_vec)
        extra = _sanitize_population(
            extra,
            dim,
            lb_vec,
            ub_vec,
            integer_encoding=integer_encoding,
        )
        pop = np.vstack([pop, extra])
        pop = np.unique(pop, axis=0)

    return pop[:target_size]


def _merge_seed_and_random(seed_pop, random_pop, target_size, dim, lb_vec, ub_vec, integer_encoding=False):
    seed_pop = _sanitize_population(
        seed_pop,
        dim,
        lb_vec,
        ub_vec,
        integer_encoding=integer_encoding,
    )
    random_pop = _sanitize_population(
        random_pop,
        dim,
        lb_vec,
        ub_vec,
        integer_encoding=integer_encoding,
    )

    merged = np.vstack([seed_pop, random_pop]) if seed_pop.shape[0] > 0 else random_pop
    merged = np.unique(merged, axis=0)
    merged = _fill_population_to_size(
        merged,
        target_size=target_size,
        dim=dim,
        lb_vec=lb_vec,
        ub_vec=ub_vec,
        integer_encoding=integer_encoding,
    )
    return merged[:target_size]


def _expand_bound(b, dim):
    arr = np.asarray(b, dtype=float).reshape(-1)
    if arr.size == 1:
        return np.full(dim, float(arr[0]), dtype=float)
    if arr.size != dim:
        raise ValueError("lb/ub must be scalar or vector with length == dim.")
    return arr


def _filter_metric_points(points, invalid_penalty):
    points = np.asarray(points, dtype=float)
    if points.size == 0:
        return np.empty((0, 0), dtype=float)
    if points.ndim == 1:
        points = points.reshape(1, -1)

    finite_mask = np.all(np.isfinite(points), axis=1)
    feasible_mask = np.all(points < float(invalid_penalty), axis=1)
    out = points[finite_mask & feasible_mask]
    if out.shape[0] == 0:
        return np.empty((0, points.shape[1]), dtype=float)
    return np.unique(out, axis=0)


def _safe_igd(valid_fit, valid_pf):
    if valid_fit.shape[0] == 0 or valid_pf.shape[0] == 0:
        return np.inf
    return float(IGD(valid_fit, valid_pf))


def _safe_gd(valid_fit, valid_pf):
    if valid_fit.shape[0] == 0 or valid_pf.shape[0] == 0:
        return np.inf
    return float(GD(valid_fit, valid_pf))


def _safe_spacing(valid_fit, valid_pf):
    if valid_fit.shape[0] <= 1:
        return 0.0
    return float(Spacing(valid_fit, valid_pf))


def _safe_spread(valid_fit, valid_pf):
    if valid_fit.shape[0] <= 1 or valid_pf.shape[0] == 0:
        return 0.0
    if valid_fit.shape[0] <= valid_fit.shape[1]:
        return 0.0
    return float(Spread(valid_fit, valid_pf))


def _safe_hv(valid_fit, valid_pf):
    if valid_fit.shape[0] == 0 or valid_pf.shape[0] == 0:
        return 0.0
    hv_out = HV(valid_fit, valid_pf)
    return float(hv_out[0] if isinstance(hv_out, tuple) else hv_out)


def _safe_coverage(valid_fit, valid_pf):
    if valid_fit.shape[0] == 0 or valid_pf.shape[0] == 0:
        return 0.0
    return float(Coverage(valid_fit, valid_pf))


def _extract_feasible_first_front_archive(archive, archive_fit, invalid_penalty):
    archive = np.asarray(archive, dtype=float)
    archive_fit = np.asarray(archive_fit, dtype=float)

    if archive.size == 0 or archive_fit.size == 0:
        return np.empty((0, 0), dtype=float), np.empty((0, 0), dtype=float)

    feasible_mask = np.all(np.isfinite(archive_fit), axis=1) & np.all(archive_fit < float(invalid_penalty), axis=1)
    if not np.any(feasible_mask):
        return np.empty((0, archive.shape[1]), dtype=float), np.empty((0, archive_fit.shape[1]), dtype=float)

    pop = archive[feasible_mask]
    fit = archive_fit[feasible_mask]
    nd_mask = np.ones(fit.shape[0], dtype=bool)

    for i in range(fit.shape[0]):
        if not nd_mask[i]:
            continue
        for j in range(fit.shape[0]):
            if i == j or not nd_mask[i]:
                continue
            if np.all(fit[j] <= fit[i]) and np.any(fit[j] < fit[i]):
                nd_mask[i] = False
                break

    return pop[nd_mask], fit[nd_mask]
