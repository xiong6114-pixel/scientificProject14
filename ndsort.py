import numpy as np


def NDSort(*args):
    """
    Python migration of MATLAB NDSort.m.

    Usage:
    - FrontNo, MaxFNo = NDSort(PopObj, nSort)
    - FrontNo, MaxFNo = NDSort(PopObj, PopCon, nSort)
    """
    if len(args) == 2:
        pop_obj = np.asarray(args[0], dtype=float)
        n_sort = args[1]
    elif len(args) == 3:
        pop_obj = np.asarray(args[0], dtype=float).copy()
        pop_con = np.asarray(args[1], dtype=float)
        n_sort = args[2]

        infeasible = np.any(pop_con > 0, axis=1)
        if np.any(infeasible):
            _, m = pop_obj.shape
            worst = np.max(pop_obj, axis=0)
            violation = np.sum(np.maximum(0.0, pop_con[infeasible, :]), axis=1)
            pop_obj[infeasible, :] = np.tile(worst, (np.sum(infeasible), 1)) + np.tile(violation[:, None], (1, m))
    else:
        raise TypeError("NDSort expects 2 or 3 input arguments.")

    if pop_obj.ndim != 2:
        raise ValueError("PopObj must be a 2D array.")

    n, m = pop_obj.shape
    if m < 3 or n < 500:
        front_no, max_f_no = _ENS_SS(pop_obj, n_sort)
    else:
        front_no, max_f_no = _T_ENS(pop_obj, n_sort)
    return front_no, max_f_no


def _ENS_SS(pop_obj, n_sort):
    # Keep MATLAB logic: unique(PopObj,'rows') + Loc remapping.
    pop_unique, loc = np.unique(pop_obj, axis=0, return_inverse=True)
    table = np.bincount(loc, minlength=pop_unique.shape[0]).astype(float)

    n, m = pop_unique.shape
    front_no = np.full(n, np.inf, dtype=float)
    max_f_no = 0
    target = min(float(n_sort), float(loc.size))

    while np.sum(table[np.isfinite(front_no)]) < target:
        max_f_no += 1
        for i in range(n):
            if np.isfinite(front_no[i]):
                continue

            dominated = False
            for j in range(i - 1, -1, -1):
                if front_no[j] == max_f_no:
                    k = 1  # MATLAB index m=2 in 1-based indexing.
                    while k < m and pop_unique[i, k] >= pop_unique[j, k]:
                        k += 1
                    dominated = k >= m
                    if dominated or m == 2:
                        break

            if not dominated:
                front_no[i] = max_f_no

    # Loc back-fill: FrontNo = FrontNo(:,Loc) in MATLAB.
    front_no = front_no[loc]
    return front_no, max_f_no


def _T_ENS(pop_obj, n_sort):
    # Keep MATLAB logic: unique(PopObj,'rows') + Loc remapping.
    pop_unique, loc = np.unique(pop_obj, axis=0, return_inverse=True)
    table = np.bincount(loc, minlength=pop_unique.shape[0]).astype(float)

    n, m = pop_unique.shape
    # Use 1-based indexing containers to stay close to MATLAB code.
    pop = np.vstack([np.zeros((1, m), dtype=float), pop_unique])

    front_no = np.full(n + 1, np.inf, dtype=float)
    max_f_no = 0
    forest = np.zeros(n + 1, dtype=int)
    children = np.zeros((n + 1, m + 1), dtype=int)  # columns 1..m-1 are used
    left_child = np.full(n + 1, m, dtype=int)
    father = np.zeros(n + 1, dtype=int)
    brother = np.full(n + 1, m, dtype=int)

    # MATLAB:
    # [~,ORank] = sort(PopObj(:,2:M),2,'descend');
    # ORank     = ORank + 1;
    o_rank = np.argsort(pop_unique[:, 1:], axis=1)[:, ::-1] + 2  # objective ids in [2..M]
    o_rank = np.vstack([np.zeros((1, m - 1), dtype=int), o_rank])

    target = min(float(n_sort), float(loc.size))
    while np.sum(table[np.isfinite(front_no[1:])]) < target:
        max_f_no += 1

        inf_idx = np.where(np.isinf(front_no[1:]))[0]
        if inf_idx.size == 0:
            break
        root = int(inf_idx[0] + 1)
        forest[max_f_no] = root
        front_no[root] = max_f_no

        for p in range(1, n + 1):
            if not np.isinf(front_no[p]):
                continue

            pruning = np.zeros(n + 1, dtype=int)
            q = forest[max_f_no]

            while True:
                k = 1
                while k < m and pop[p, o_rank[q, k - 1] - 1] >= pop[q, o_rank[q, k - 1] - 1]:
                    k += 1

                if k == m:
                    break

                pruning[q] = k
                if left_child[q] <= pruning[q]:
                    q = children[q, left_child[q]]
                else:
                    while father[q] != 0 and brother[q] > pruning[father[q]]:
                        q = father[q]
                    if father[q] != 0:
                        q = children[father[q], brother[q]]
                    else:
                        break

            if k < m:
                front_no[p] = max_f_no
                q = forest[max_f_no]
                while children[q, pruning[q]] != 0:
                    q = children[q, pruning[q]]

                children[q, pruning[q]] = p
                father[p] = q

                if left_child[q] > pruning[q]:
                    brother[p] = left_child[q]
                    left_child[q] = pruning[q]
                else:
                    bro = children[q, left_child[q]]
                    while brother[bro] < pruning[q]:
                        bro = children[q, brother[bro]]
                    brother[p] = brother[bro]
                    brother[bro] = pruning[q]

    # Loc back-fill: FrontNo = FrontNo(:,Loc) in MATLAB.
    front_no = front_no[1:][loc]
    return front_no, max_f_no


if __name__ == "__main__":
    # Small 2D sample
    pop_obj = np.array(
        [
            [0.10, 0.90],
            [0.20, 0.80],
            [0.30, 0.70],
            [0.40, 0.60],
            [0.20, 0.95],
            [0.90, 0.20],
            [0.60, 0.60],
        ],
        dtype=float,
    )
    front_no, max_f_no = NDSort(pop_obj, np.inf)
    print("PopObj:")
    print(pop_obj)
    print("FrontNo:", front_no)
    print("MaxFNo:", max_f_no)
