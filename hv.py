import numpy as np


def HV(PopObj, PF):
    """
    Python migration of MATLAB HV.m.

    Returns
    -------
    Score : float
    PopObj : np.ndarray
        Normalized and filtered objective matrix used by HV.
    """
    pop_obj = np.asarray(PopObj, dtype=float)
    pf = np.asarray(PF, dtype=float)

    if pop_obj.ndim != 2 or pf.ndim != 2:
        raise ValueError("PopObj and PF must both be 2D arrays.")
    if pop_obj.shape[1] != pf.shape[1]:
        raise ValueError("PopObj and PF must have the same number of columns.")

    n, m = pop_obj.shape
    fmin = np.minimum(np.min(pop_obj, axis=0), np.zeros(m))
    fmax = np.max(pf, axis=0)
    pop_obj = (pop_obj - np.tile(fmin, (n, 1))) / np.tile((fmax - fmin) * 1.1, (n, 1))
    pop_obj = pop_obj[~np.any(pop_obj > 1, axis=1), :]

    # The reference point is set to (1,1,...)
    ref_point = np.ones(m)

    if pop_obj.size == 0:
        score = 0.0
    elif m < 4:
        # Calculate the exact HV value
        pl = _sortrows(pop_obj)
        S = [(1.0, pl)]
        for k in range(1, m):
            S_ = []
            for coef, mat in S:
                Stemp = Slice(mat, k, ref_point)
                for coef2, mat2 in Stemp:
                    temp = (float(coef2) * float(coef), mat2)
                    S_ = Add(temp, S_)
            S = S_

        score = 0.0
        for coef, mat in S:
            p = Head(mat)
            score = score + float(coef) * abs(p[m - 1] - ref_point[m - 1])
    else:
        # Estimate the HV value by Monte Carlo estimation
        # Note: this keeps the original default SampleNum for behavior parity.
        # Potential optimization direction: chunked sampling to reduce peak memory.
        sample_num = 1000000
        max_value = ref_point
        min_value = np.min(pop_obj, axis=0)
        samples = np.random.uniform(low=min_value, high=max_value, size=(sample_num, m))

        for i in range(pop_obj.shape[0]):
            domi = np.ones(samples.shape[0], dtype=bool)
            mm = 0
            while mm < m and np.any(domi):
                domi = domi & (pop_obj[i, mm] <= samples[:, mm])
                mm = mm + 1
            samples = samples[~domi, :]

        score = np.prod(max_value - min_value) * (1 - samples.shape[0] / sample_num)

    return float(score), pop_obj


def Slice(pl, k, RefPoint):
    p = Head(pl)
    pl = Tail(pl)
    ql = np.empty((0, 0), dtype=float)
    S = []

    while not _isempty(pl):
        ql = Insert(p, k + 1, ql)
        p_ = Head(pl)
        cell_ = (abs(p[k - 1] - p_[k - 1]), ql)
        S = Add(cell_, S)
        p = p_
        pl = Tail(pl)

    ql = Insert(p, k + 1, ql)
    cell_ = (abs(p[k - 1] - RefPoint[k - 1]), ql)
    S = Add(cell_, S)
    return S


def Insert(p, k, pl):
    flag1 = 0
    flag2 = 0
    ql = np.empty((0, p.shape[0]), dtype=float)

    hp = Head(pl)
    while (not _isempty(pl)) and (hp[k - 1] < p[k - 1]):
        ql = np.vstack([ql, hp])
        pl = Tail(pl)
        hp = Head(pl)

    ql = np.vstack([ql, p])
    m = p.shape[0]

    while not _isempty(pl):
        q = Head(pl)
        for i in range(k - 1, m):
            if p[i] < q[i]:
                flag1 = 1
            else:
                if p[i] > q[i]:
                    flag2 = 1
        if not (flag1 == 1 and flag2 == 0):
            ql = np.vstack([ql, Head(pl)])
        pl = Tail(pl)
    return ql


def Head(pl):
    if _isempty(pl):
        return np.array([], dtype=float)
    return pl[0, :]


def Tail(pl):
    if _isempty(pl) or pl.shape[0] < 2:
        return np.empty((0, pl.shape[1] if pl.ndim == 2 else 0), dtype=float)
    return pl[1:, :]


def Add(cell_, S):
    n = len(S)
    found = 0
    for k in range(n):
        if _isequal(cell_[1], S[k][1]):
            S[k] = (float(S[k][0]) + float(cell_[0]), S[k][1])
            found = 1
            break
    if found == 0:
        S.append((float(cell_[0]), np.array(cell_[1], copy=True)))
    return S


def _sortrows(x):
    if x.size == 0:
        return x
    keys = tuple(x[:, i] for i in range(x.shape[1] - 1, -1, -1))
    idx = np.lexsort(keys)
    return x[idx, :]


def _isempty(a):
    return a.size == 0


def _isequal(a, b):
    return np.array_equal(a, b)


if __name__ == "__main__":
    # 2D test sample (exact branch M<4), and print normalized PopObj for MATLAB comparison.
    pop_obj = np.array(
        [
            [0.2, 0.8],
            [0.4, 0.6],
            [0.7, 0.3],
            [1.2, 0.1],
        ],
        dtype=float,
    )
    pf = np.array(
        [
            [0.0, 1.0],
            [1.0, 0.0],
        ],
        dtype=float,
    )

    score, pop_obj_norm = HV(pop_obj, pf)
    print("Normalized PopObj:")
    print(pop_obj_norm)
    print("HV Score:", score)
