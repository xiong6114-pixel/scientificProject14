import numpy as np

from get_mofcn import getMOFcn


def EnvironmentalSelection(FUN, Population, N, M, Z, Zmin):
    """
    Python translation of MATLAB EnvironmentalSelection.m

    Parameters
    ----------
    FUN : str
        Problem name / function flag passed to getMOFcn
    Population : np.ndarray, shape=(P, D)
    N : int
        Target population size
    M : int
        Number of objectives
    Z : np.ndarray, shape=(NZ, M)
        Reference points
    Zmin : np.ndarray, shape=(M,) or None
        Ideal point

    Returns
    -------
    Population : np.ndarray, shape=(N, D)
        Selected population
    """
    Population = np.asarray(Population, dtype=float)
    Z = np.asarray(Z, dtype=float)

    if Zmin is None or len(np.asarray(Zmin).reshape(-1)) == 0:
        Zmin = np.ones(Z.shape[1], dtype=float)
    else:
        Zmin = np.asarray(Zmin, dtype=float).reshape(-1)

    # Non-dominated sorting
    Population_objs, _pop_con, _ = getMOFcn(FUN, Population, M)
    FrontNo, MaxFNo = NDSort(Population_objs, N)

    Next = FrontNo < MaxFNo

    # Select the solutions in the last front
    Last = np.where(FrontNo == MaxFNo)[0]
    K = int(N - np.sum(Next))

    if K > 0 and Last.size > 0:
        Choose = LastSelection(
            Population_objs[Next, :],
            Population_objs[Last, :],
            K,
            Z,
            Zmin,
        )
        Next[Last[Choose]] = True

    Population = Population[Next, :]
    return Population


def LastSelection(PopObj1, PopObj2, K, Z, Zmin):
    """
    Select part of the solutions in the last front.
    Returns a boolean vector of length len(PopObj2).
    """
    PopObj1 = np.asarray(PopObj1, dtype=float)
    PopObj2 = np.asarray(PopObj2, dtype=float)
    Z = np.asarray(Z, dtype=float)
    Zmin = np.asarray(Zmin, dtype=float).reshape(-1)

    if PopObj2.shape[0] == 0 or K <= 0:
        return np.zeros(PopObj2.shape[0], dtype=bool)

    PopObj = np.vstack([PopObj1, PopObj2]) - Zmin.reshape(1, -1)

    N, M = PopObj.shape
    N1 = PopObj1.shape[0]
    N2 = PopObj2.shape[0]
    NZ = Z.shape[0]

    # ---------- Normalization ----------
    Extreme = np.zeros(M, dtype=int)
    w = np.eye(M) + 1e-6

    for i in range(M):
        # MATLAB:
        # [~,Extreme(i)] = min(max(PopObj./repmat(w(i,:),N,1),[],2));
        val = np.max(PopObj / w[i, :].reshape(1, -1), axis=1)
        Extreme[i] = int(np.argmin(val))

    # Hyperplane intercepts
    try:
        Hyperplane = np.linalg.solve(PopObj[Extreme, :], np.ones(M))
        a = 1.0 / Hyperplane
        if np.any(~np.isfinite(a)):
            raise np.linalg.LinAlgError
    except np.linalg.LinAlgError:
        a = np.max(PopObj, axis=0)

    a = np.asarray(a, dtype=float).reshape(-1)
    a[~np.isfinite(a)] = 1.0
    a[np.abs(a) < 1e-12] = 1.0

    PopObj = PopObj / a.reshape(1, -1)

    # ---------- Associate each solution with one reference point ----------
    # cosine similarity
    pop_norm = np.linalg.norm(PopObj, axis=1, keepdims=True)
    z_norm = np.linalg.norm(Z, axis=1, keepdims=True).T

    pop_norm = np.maximum(pop_norm, 1e-12)
    z_norm = np.maximum(z_norm, 1e-12)

    Cosine = (PopObj @ Z.T) / (pop_norm * z_norm)
    Cosine = np.clip(Cosine, -1.0, 1.0)

    # MATLAB:
    # Distance = repmat(sqrt(sum(PopObj.^2,2)),1,NZ).*sqrt(1-Cosine.^2);
    Distance = pop_norm * np.sqrt(np.maximum(0.0, 1.0 - Cosine ** 2))

    # nearest reference point
    pi = np.argmin(Distance, axis=1)        # 0-based
    d = Distance[np.arange(N), pi]

    # ---------- rho: associated solutions except last front ----------
    rho = np.bincount(pi[:N1], minlength=NZ).astype(int)

    # ---------- Environmental selection ----------
    Choose = np.zeros(N2, dtype=bool)
    Zchoose = np.ones(NZ, dtype=bool)

    while np.sum(Choose) < K:
        Temp = np.where(Zchoose)[0]
        if Temp.size == 0:
            break

        min_rho = np.min(rho[Temp])
        Jmin = Temp[rho[Temp] == min_rho]
        j = int(Jmin[np.random.randint(len(Jmin))])

        # indices in PopObj2 associated with ref point j and not yet chosen
        I = np.where((~Choose) & (pi[N1:] == j))[0]

        if I.size > 0:
            if rho[j] == 0:
                s = int(np.argmin(d[N1 + I]))
            else:
                s = int(np.random.randint(I.size))

            Choose[I[s]] = True
            rho[j] += 1
        else:
            Zchoose[j] = False

    return Choose


def NDSort(PopObj, N):
    """
    Simple non-dominated sorting.
    Returns
    -------
    FrontNo : np.ndarray, shape=(P,)
    MaxFNo : int
    """
    PopObj = np.asarray(PopObj, dtype=float)
    P = PopObj.shape[0]

    FrontNo = np.full(P, np.inf)
    dominate_me = np.zeros(P, dtype=int)
    i_dominate = [[] for _ in range(P)]

    for p in range(P):
        for q in range(p + 1, P):
            p_dom_q = np.all(PopObj[p] <= PopObj[q]) and np.any(PopObj[p] < PopObj[q])
            q_dom_p = np.all(PopObj[q] <= PopObj[p]) and np.any(PopObj[q] < PopObj[p])

            if p_dom_q:
                i_dominate[p].append(q)
                dominate_me[q] += 1
            elif q_dom_p:
                i_dominate[q].append(p)
                dominate_me[p] += 1

    current_front = np.where(dominate_me == 0)[0]
    front = 1
    assigned = 0

    while current_front.size > 0 and assigned < min(N, P):
        FrontNo[current_front] = front
        assigned += current_front.size

        next_front = []
        for p in current_front:
            for q in i_dominate[p]:
                dominate_me[q] -= 1
                if dominate_me[q] == 0:
                    next_front.append(q)

        current_front = np.array(next_front, dtype=int)
        front += 1

    MaxFNo = int(np.nanmax(FrontNo[np.isfinite(FrontNo)]))
    FrontNo[~np.isfinite(FrontNo)] = MaxFNo + 1
    return FrontNo.astype(int), MaxFNo