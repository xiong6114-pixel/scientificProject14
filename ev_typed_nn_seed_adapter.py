import numpy as np

from cal_obj_typed_py import diagnose_cal_obj_typed


def split_total_into_types(total_piles, nmax_vec, type_prob, stochastic=False):
    total_piles = int(round(total_piles))
    nmax_vec = np.asarray(nmax_vec, dtype=int).reshape(3)
    type_prob = np.asarray(type_prob, dtype=float).reshape(3)
    type_prob = type_prob / np.sum(type_prob)

    if total_piles <= 0:
        return np.array([0, 0, 0], dtype=int)

    total_cap = int(np.sum(nmax_vec))
    total_piles = min(total_piles, total_cap)

    if stochastic:
        counts = np.random.multinomial(total_piles, type_prob)
    else:
        raw = total_piles * type_prob
        counts = np.floor(raw).astype(int)
        remain = total_piles - counts.sum()

        if remain > 0:
            frac = raw - counts
            order = np.argsort(-frac)
            for k in order[:remain]:
                counts[k] += 1

    counts = np.minimum(counts, nmax_vec)

    remain = total_piles - int(np.sum(counts))
    while remain > 0:
        cap = nmax_vec - counts
        idx = np.where(cap > 0)[0]
        if len(idx) == 0:
            break

        if stochastic:
            kk = int(np.random.choice(idx))
        else:
            kk = int(idx[np.argmax(cap[idx])])

        counts[kk] += 1
        remain -= 1

    if counts.sum() == 0 and total_piles > 0:
        idx = np.where(nmax_vec > 0)[0]
        if len(idx) > 0:
            counts[int(idx[0])] = 1

    return counts.astype(int)


def typed_pack_solution(build_vec, piles_vec, level_vec=None, context=None):
    """
    网络输出:
        build_vec: [100]
        piles_vec: [100]  total piles at each candidate point

    目标 typed 染色体:
        sol = [nSlow(1..M), nFast(1..M), nUltra(1..M)]
    """
    context = context or {}

    M = int(context["charge_points_num"])
    type_prob = np.asarray(context["type_prob"], dtype=float).reshape(3)
    type_nmax = np.asarray(context["type_nmax"], dtype=int).reshape(3)
    stochastic_split = bool(context.get("stochastic_split", False))

    build_vec = np.asarray(build_vec).reshape(-1)
    piles_vec = np.asarray(piles_vec).reshape(-1)

    build = build_vec[:M].astype(int)
    total_piles = np.round(piles_vec[:M]).astype(int)
    total_piles[build <= 0] = 0
    total_piles[total_piles < 0] = 0

    nSlow = np.zeros(M, dtype=int)
    nFast = np.zeros(M, dtype=int)
    nUltra = np.zeros(M, dtype=int)

    for j in range(M):
        if total_piles[j] <= 0:
            continue

        counts = split_total_into_types(
            total_piles=int(total_piles[j]),
            nmax_vec=type_nmax,
            type_prob=type_prob,
            stochastic=stochastic_split,
        )
        nSlow[j], nFast[j], nUltra[j] = counts.tolist()

    sol = np.concatenate([nSlow, nFast, nUltra], axis=0).astype(float)
    return sol


def typed_repair_solution(sol, context=None):
    context = context or {}

    M = int(context["charge_points_num"])
    type_nmax = np.asarray(context["type_nmax"], dtype=int).reshape(3)
    fallback_index = int(context.get("fallback_index", 0))

    sol = np.asarray(sol, dtype=float).reshape(-1).copy()
    if sol.shape[0] != 3 * M:
        raise ValueError(f"typed sol 长度应为 {3*M}，实际 {sol.shape[0]}")

    nSlow = np.round(sol[:M]).astype(int)
    nFast = np.round(sol[M:2*M]).astype(int)
    nUltra = np.round(sol[2*M:3*M]).astype(int)

    nSlow = np.clip(nSlow, 0, type_nmax[0])
    nFast = np.clip(nFast, 0, type_nmax[1])
    nUltra = np.clip(nUltra, 0, type_nmax[2])

    total = nSlow + nFast + nUltra
    if np.all(total <= 0):
        j = max(0, min(fallback_index, M - 1))
        if type_nmax[1] > 0:
            nFast[j] = 1
        elif type_nmax[0] > 0:
            nSlow[j] = 1
        elif type_nmax[2] > 0:
            nUltra[j] = 1

    sol = np.concatenate([nSlow, nFast, nUltra], axis=0).astype(float)

    if not bool(context.get("enable_seed_feasibility_repair", True)):
        return sol

    required_keys = (
        "demand_points_info",
        "charge_points_info",
        "parameter",
        "typeInfo",
    )
    if any(k not in context for k in required_keys):
        return sol

    return _repair_typed_solution_to_feasible(sol, context)


def typed_evaluate_solution(sol, context=None):
    context = context or {}

    cal_obj_fn = context["cal_obj_fn"]
    demand_points_info = context["demand_points_info"]
    charge_points_info = context["charge_points_info"]
    parameter = context["parameter"]
    type_info = context["typeInfo"]
    dist_matrix = context.get("dist_matrix", None)

    f = cal_obj_fn(
        sol,
        demand_points_info,
        charge_points_info,
        parameter,
        type_info,
        dist_matrix=dist_matrix,
    )
    f = np.asarray(f, dtype=float).reshape(-1)

    if f.size != 2:
        return None
    if not np.all(np.isfinite(f)):
        return None
    if np.any(f == -1):
        return None

    return f


def _repair_typed_solution_to_feasible(sol, context):
    M = int(context["charge_points_num"])
    type_nmax = np.asarray(context["type_nmax"], dtype=int).reshape(3)
    type_info = context["typeInfo"]
    demand_points_info = context["demand_points_info"]
    charge_points_info = context["charge_points_info"]
    parameter = context["parameter"]
    dist_matrix = context.get("dist_matrix", None)

    sol = np.asarray(sol, dtype=float).reshape(-1).copy()
    max_passes = int(context.get("seed_repair_max_passes", 6))
    service_margin = float(context.get("seed_repair_service_margin", 1.02))

    for _ in range(max_passes):
        analysis = diagnose_cal_obj_typed(
            sol,
            demand_points_info,
            charge_points_info,
            parameter=parameter,
            type_info=type_info,
            dist_matrix=dist_matrix,
            print_fn=None,
        )
        if analysis["feasible"]:
            return sol

        changed = False
        n_slow = np.round(sol[:M]).astype(int)
        n_fast = np.round(sol[M : 2 * M]).astype(int)
        n_ultra = np.round(sol[2 * M : 3 * M]).astype(int)

        if analysis["unreachable_demand_idx"].size > 0 and dist_matrix is not None:
            changed |= _activate_unreachable_nearest_stations(
                n_slow=n_slow,
                n_fast=n_fast,
                n_ultra=n_ultra,
                unreachable_idx=np.asarray(analysis["unreachable_demand_idx"], dtype=int),
                dist_matrix=np.asarray(dist_matrix, dtype=float),
                type_nmax=type_nmax,
            )

        if analysis["station_table"].size > 0:
            for row in np.asarray(analysis["station_table"], dtype=float):
                j = int(row[0])
                lam = float(row[1])
                code = int(row[6])

                if code == 1:
                    if (n_slow[j] + n_fast[j] + n_ultra[j]) > 0:
                        n_slow[j] = 0
                        n_fast[j] = 0
                        n_ultra[j] = 0
                        changed = True
                elif code in (4, 5):
                    changed |= _promote_station_to_service_target(
                        n_slow=n_slow,
                        n_fast=n_fast,
                        n_ultra=n_ultra,
                        station_idx=j,
                        target_lambda=lam,
                        type_nmax=type_nmax,
                        type_info=type_info,
                        service_margin=service_margin,
                    )

        if not changed:
            break

        sol = np.concatenate([n_slow, n_fast, n_ultra], axis=0).astype(float)

    return sol


def _activate_unreachable_nearest_stations(
    n_slow,
    n_fast,
    n_ultra,
    unreachable_idx,
    dist_matrix,
    type_nmax,
):
    changed = False
    for i in unreachable_idx:
        j = int(np.argmin(dist_matrix[i]))
        if (n_slow[j] + n_fast[j] + n_ultra[j]) > 0:
            continue

        if type_nmax[1] > 0:
            n_fast[j] = max(n_fast[j], 1)
        elif type_nmax[0] > 0:
            n_slow[j] = max(n_slow[j], 1)
        elif type_nmax[2] > 0:
            n_ultra[j] = max(n_ultra[j], 1)
        else:
            continue
        changed = True
    return changed


def _promote_station_to_service_target(
    n_slow,
    n_fast,
    n_ultra,
    station_idx,
    target_lambda,
    type_nmax,
    type_info,
    service_margin,
):
    j = int(station_idx)
    mu_vec = np.asarray(type_info.mu, dtype=float).reshape(3)
    target_service = float(target_lambda) * max(float(service_margin), 1.0)

    changed = False
    while True:
        current_service = (
            n_slow[j] * mu_vec[0]
            + n_fast[j] * mu_vec[1]
            + n_ultra[j] * mu_vec[2]
        )
        if current_service >= target_service:
            break

        if n_ultra[j] < type_nmax[2]:
            n_ultra[j] += 1
        elif n_fast[j] < type_nmax[1]:
            n_fast[j] += 1
        elif n_slow[j] < type_nmax[0]:
            n_slow[j] += 1
        else:
            break
        changed = True

    return changed


def build_ev_typed_problem_context(
    x_raw_flat,
    charge_points_num,
    cal_obj_fn,
    demand_points_info,
    charge_points_info,
    parameter,
    typeInfo,
    type_prob=(0.50, 0.35, 0.15),
    fallback_index=0,
    stochastic_split=False,
    dist_matrix=None,
    invalid_penalty=1e12,
    allow_infeasible_seed_fallback=True,
    enable_seed_feasibility_repair=True,
    seed_repair_service_margin=1.02,
    seed_repair_max_passes=6,
    debug_seed_injection=True,
    debug_metrics=True,
):
    type_nmax = np.asarray(typeInfo.nmax, dtype=int).reshape(3)

    return {
        "x_raw_flat": np.asarray(x_raw_flat, dtype=np.float32).reshape(-1),
        "charge_points_num": int(charge_points_num),
        "typeInfo": typeInfo,
        "type_nmax": type_nmax,
        "type_prob": np.asarray(type_prob, dtype=float).reshape(3),
        "fallback_index": int(fallback_index),
        "stochastic_split": bool(stochastic_split),

        "cal_obj_fn": cal_obj_fn,
        "demand_points_info": demand_points_info,
        "charge_points_info": charge_points_info,
        "parameter": parameter,
        "dist_matrix": dist_matrix,
        "invalid_penalty": float(invalid_penalty),

        "pack_solution_fn": typed_pack_solution,
        "repair_fn": typed_repair_solution,
        "evaluate_fn": typed_evaluate_solution,

        "allow_infeasible_seed_fallback": bool(allow_infeasible_seed_fallback),
        "enable_seed_feasibility_repair": bool(enable_seed_feasibility_repair),
        "seed_repair_service_margin": float(seed_repair_service_margin),
        "seed_repair_max_passes": int(seed_repair_max_passes),
        "debug_seed_injection": bool(debug_seed_injection),
        "debug_metrics": bool(debug_metrics),
        "integer_encoding": True,
    }
