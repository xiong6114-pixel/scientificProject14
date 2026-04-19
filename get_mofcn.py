import numpy as np

_GLOBAL_EV_CONTEXT = None


def set_ev_context(context: dict):
    global _GLOBAL_EV_CONTEXT
    _GLOBAL_EV_CONTEXT = context


def clear_ev_context():
    global _GLOBAL_EV_CONTEXT
    _GLOBAL_EV_CONTEXT = None


def getMOFcn(F, PopDec, numObj):
    PopDec = np.asarray(PopDec, dtype=float)
    if PopDec.ndim != 2:
        raise ValueError("PopDec must be a 2D array")

    N, _ = PopDec.shape
    PopCon = np.empty((N, 0), dtype=float)

    if N == 0:
        return np.empty((0, numObj), dtype=float), PopCon, np.empty((0, numObj), dtype=float)

    if F in {"ZDT1", "ZDT2", "ZDT3", "ZDT4"}:
        prob_id = int(F[-1])
        PopObj = np.vstack([_zdt_cost(PopDec[i, :], prob_id) for i in range(N)]).astype(float)
        P = _zdt_true_pf(prob_id, 1000)
        return PopObj, PopCon, P

    if F == "EV_TYPED_CS":
        if _GLOBAL_EV_CONTEXT is None:
            raise RuntimeError("EV_TYPED_CS requires set_ev_context(context) before evaluation")

        cal_obj_fn = _GLOBAL_EV_CONTEXT["cal_obj_fn"]
        demand_points_info = _GLOBAL_EV_CONTEXT["demand_points_info"]
        charge_points_info = _GLOBAL_EV_CONTEXT["charge_points_info"]
        parameter = _GLOBAL_EV_CONTEXT["parameter"]
        type_info = _GLOBAL_EV_CONTEXT["typeInfo"]
        dist_matrix = _GLOBAL_EV_CONTEXT.get("dist_matrix", None)
        invalid_penalty = float(_GLOBAL_EV_CONTEXT.get("invalid_penalty", 1e12))

        PopObj = np.zeros((N, numObj), dtype=float)
        feasible_mask = np.ones(N, dtype=bool)

        for i in range(N):
            sol = np.round(PopDec[i, :])
            f = cal_obj_fn(
                sol,
                demand_points_info,
                charge_points_info,
                parameter,
                type_info,
                dist_matrix=dist_matrix,
            )
            f = np.asarray(f, dtype=float).reshape(-1)

            if f.size != numObj:
                raise ValueError(f"Expected {numObj} objectives, got {f.shape}")

            if (not np.all(np.isfinite(f))) or np.any(f == -1):
                f = np.full(numObj, invalid_penalty, dtype=float)
                feasible_mask[i] = False

            PopObj[i, :] = f

        front_no = ndsort_first_front(PopObj)
        pf_mask = (front_no == 1) & feasible_mask
        P = PopObj[pf_mask, :]
        return PopObj, PopCon, P

    raise NotImplementedError(f"Unsupported getMOFcn problem: {F}")


def ndsort_first_front(PopObj):
    N = PopObj.shape[0]
    if N == 0:
        return np.empty((0,), dtype=int)

    front_no = np.full(N, np.inf)
    dominated_count = np.zeros(N, dtype=int)
    dominates = [[] for _ in range(N)]

    for i in range(N):
        for j in range(i + 1, N):
            i_dom_j = np.all(PopObj[i] <= PopObj[j]) and np.any(PopObj[i] < PopObj[j])
            j_dom_i = np.all(PopObj[j] <= PopObj[i]) and np.any(PopObj[j] < PopObj[i])

            if i_dom_j:
                dominates[i].append(j)
                dominated_count[j] += 1
            elif j_dom_i:
                dominates[j].append(i)
                dominated_count[i] += 1

    front = np.where(dominated_count == 0)[0]
    front_no[front] = 1
    return front_no


def _zdt_cost(x, prob_id):
    x = np.asarray(x, dtype=float).reshape(-1)
    d = x.size

    if d < 2:
        raise ValueError("ZDT decision vector must have at least 2 variables.")

    f1 = float(x[0])
    if prob_id == 1:
        g = float(1.0 + 9.0 * np.sum(x[1:]) / (d - 1))
        f2 = float(g * (1.0 - np.sqrt(f1 / g)))
    elif prob_id == 2:
        g = float(1.0 + 9.0 * np.sum(x[1:]) / (d - 1))
        f2 = float(g * (1.0 - (f1 / g) ** 2))
    elif prob_id == 3:
        g = float(1.0 + 9.0 * np.sum(x[1:]) / (d - 1))
        ratio = f1 / g
        f2 = float(g * (1.0 - np.sqrt(ratio) - ratio * np.sin(10.0 * np.pi * f1)))
    elif prob_id == 4:
        g = float(1.0 + 10.0 * (d - 1) + np.sum(x[1:] ** 2 - 10.0 * np.cos(4.0 * np.pi * x[1:])))
        f2 = float(g * (1.0 - np.sqrt(f1 / g)))
    else:
        raise ValueError(f"Unsupported ZDT problem: {prob_id}")

    return np.array([f1, f2], dtype=float)


def _zdt_true_pf(prob_id, n_points):
    x1 = np.linspace(0.0, 1.0, int(n_points), dtype=float)

    if prob_id == 1:
        f2 = 1.0 - np.sqrt(x1)
    elif prob_id == 2:
        f2 = 1.0 - x1**2
    elif prob_id == 3:
        f2 = 1.0 - np.sqrt(x1) - x1 * np.sin(10.0 * np.pi * x1)
    elif prob_id == 4:
        f2 = 1.0 - np.sqrt(x1)
    else:
        raise ValueError(f"Unsupported ZDT problem: {prob_id}")

    return np.column_stack([x1, f2]).astype(float)
