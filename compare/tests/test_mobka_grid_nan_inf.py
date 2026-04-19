from __future__ import annotations

import numpy as np

from mobka_common import Particle, cgrid, fgrid


def test_fgrid_handles_nan_inf_without_index_error() -> None:
    rep = [
        Particle(
            Position=np.array([1.0, 2.0], dtype=np.float64),
            Cost=np.array([np.inf, 10.0], dtype=np.float64),
            BestPosition=np.array([1.0, 2.0], dtype=np.float64),
            BestCost=np.array([np.inf, 10.0], dtype=np.float64),
        ),
        Particle(
            Position=np.array([2.0, 1.0], dtype=np.float64),
            Cost=np.array([np.nan, 20.0], dtype=np.float64),
            BestPosition=np.array([2.0, 1.0], dtype=np.float64),
            BestCost=np.array([np.nan, 20.0], dtype=np.float64),
        ),
    ]

    grid = cgrid(rep, n_grid=6, alpha=0.1)

    p0 = fgrid(rep[0], grid)
    p1 = fgrid(rep[1], grid)

    assert p0.GridSubIndex is not None
    assert p1.GridSubIndex is not None
    assert p0.GridSubIndex.shape == (2,)
    assert p1.GridSubIndex.shape == (2,)
    assert int(p0.GridSubIndex[0]) == 0
    assert int(p1.GridSubIndex[0]) == 0
    assert isinstance(p0.GridIndex, int)
    assert isinstance(p1.GridIndex, int)

