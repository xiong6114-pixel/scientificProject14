import numpy as np


def Spread(PopObj, PF) -> float:
    """
    MATLAB equivalent of:
        Dis1  = pdist2(PopObj,PopObj);
        Dis1(logical(eye(size(Dis1,1)))) = inf;
        [~,E] = max(PF,[],1);
        Dis2  = pdist2(PF(E,:),PopObj);
        d1    = sum(min(Dis2,[],2));
        d2    = mean(min(Dis1,[],2));
        Score = (d1+sum(abs(min(Dis1,[],2)-d2))) / (d1+(size(PopObj,1)-size(PopObj,2))*d2);
    """
    pop_obj = np.asarray(PopObj, dtype=float)
    pf = np.asarray(PF, dtype=float)

    if pop_obj.ndim != 2 or pf.ndim != 2:
        raise ValueError("PopObj and PF must both be 2D arrays.")
    if pop_obj.shape[1] != pf.shape[1]:
        raise ValueError("PopObj and PF must have the same number of columns.")
    if pop_obj.shape[0] == 0 or pf.shape[0] == 0:
        raise ValueError("PopObj and PF must be non-empty.")

    # Dis1 = pdist2(PopObj, PopObj) with Euclidean distance
    diff1 = pop_obj[:, None, :] - pop_obj[None, :, :]
    dis1 = np.sqrt(np.sum(diff1 * diff1, axis=2))

    # Set diagonal to inf
    np.fill_diagonal(dis1, np.inf)

    # [~,E] = max(PF,[],1)  -> column-wise argmax (0-based in Python)
    E = np.argmax(pf, axis=0)

    # Dis2 = pdist2(PF(E,:), PopObj) with Euclidean distance
    pf_e = pf[E, :]
    diff2 = pf_e[:, None, :] - pop_obj[None, :, :]
    dis2 = np.sqrt(np.sum(diff2 * diff2, axis=2))

    min_dis2 = np.min(dis2, axis=1)
    d1 = np.sum(min_dis2)

    min_dis1 = np.min(dis1, axis=1)
    d2 = np.mean(min_dis1)

    score = (d1 + np.sum(np.abs(min_dis1 - d2))) / (d1 + (pop_obj.shape[0] - pop_obj.shape[1]) * d2)
    return float(score)


if __name__ == "__main__":
    # Minimal test
    pop_obj = np.array(
        [
            [0.0, 0.0],
            [0.5, 0.5],
            [1.0, 1.0],
        ],
        dtype=float,
    )
    pf = np.array(
        [
            [0.0, 1.0],
            [1.0, 0.0],
            [0.8, 0.8],
        ],
        dtype=float,
    )

    score = Spread(pop_obj, pf)
    print("Score =", score)
