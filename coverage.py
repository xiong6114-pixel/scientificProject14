import numpy as np


def Coverage(PopObj, PF) -> float:
    """
    MATLAB equivalent:
        Domi = false(1,size(PopObj,1));
        for i = 1 : size(PF,1)
            Domi(sum(repmat(PF(i,:),size(PopObj,1),1)-PopObj<=0,2)==size(PopObj,2)) = true;
        end
        Score = sum(Domi) / size(PopObj,1);
    """
    pop_obj = np.asarray(PopObj, dtype=float)
    pf = np.asarray(PF, dtype=float)

    if pop_obj.ndim != 2 or pf.ndim != 2:
        raise ValueError("PopObj and PF must both be 2D arrays.")
    if pop_obj.shape[1] != pf.shape[1]:
        raise ValueError("PopObj and PF must have the same number of objectives (columns).")
    if pop_obj.shape[0] == 0:
        raise ValueError("PopObj must be non-empty.")

    domi = np.zeros(pop_obj.shape[0], dtype=bool)
    for i in range(pf.shape[0]):
        # repmat(PF(i,:),size(PopObj,1),1)-PopObj<=0
        cond = np.sum((np.tile(pf[i, :], (pop_obj.shape[0], 1)) - pop_obj) <= 0, axis=1) == pop_obj.shape[1]
        domi[cond] = True

    score = np.sum(domi) / pop_obj.shape[0]
    return float(score)

