"""
Reference results for test_dd_1x1d.py, created with the old scripts.

Run once, before dlr_dd and dlr_inflow are rewritten:
    python -m DLR_rt.tests.make_reference
"""
from pathlib import Path

import numpy as np

from DLR_rt.examples.d1x1 import dlr_dd, dlr_inflow
from DLR_rt.src.dd_1x1d import full_f, setup_domains

# Small and short, but long enough that the rank changes in both runs
NX, NMU, R, T_F, DT, TOL, SEED = 256, 32, 5, 0.2, 1e-3, 1e-5, 0
PATH = Path(__file__).parent / "data" / "reference_dd_1x1d.npz"

if __name__ == "__main__":
    grids, lr0s, rngs = setup_domains(NX, NMU, R, 2, SEED)
    lr_l, lr_r, _, rank_l, rank_r, _, _ = dlr_dd.integrate(
        lr0s[0], lr0s[1], grids[0], grids[1], T_F, DT,
        tol_sing_val=TOL, drop_tol=TOL, rng_left=rngs[0], rng_right=rngs[1],
    )

    grids, lr0s, rngs = setup_domains(NX, NMU, R, 1, SEED)
    lr, _, adapt_rank, _, _ = dlr_inflow.integrate(
        lr0s[0], grids[0], T_F, DT, tol_sing_val=TOL, drop_tol=TOL, rng=rngs[0],
    )

    PATH.parent.mkdir(exist_ok=True)
    np.savez(
        PATH,
        f_2dom=np.concatenate((full_f(lr_l), full_f(lr_r)), axis=0),
        rank_hist_2dom=np.array([rank_l, rank_r]),
        f_1dom=full_f(lr),
        rank_hist_1dom=np.array([adapt_rank]),
    )
    print(f"Saved {PATH}")
