"""
Serial cost per time step against Nx and Nmu, with 1 and 2 subdomains.

    python -m DLR_rt.examples.d1x1.benchmark_scaling

Results: plots/scaling.png, plots/scaling.npz
"""
import os

# BLAS threads to 1, has to be set before numpy is imported
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from DLR_rt.src.dd_1x1d import integrate_serial, setup_domains

T_F = 0.05  # short runs, we only compare
CFL = 0.25  # dt = CFL * dx
N_REPEAT = 3  # take the fastest run
R0 = 5
SEED = 0


def time_per_step(Nx, Nmu=128, tol=1e-5, n_domains=1):
    """Fastest time per step [ms] over N_REPEAT runs."""
    best = np.inf
    for _ in range(N_REPEAT):
        grids, lr0s, rngs = setup_domains(Nx, Nmu, R0, n_domains, SEED)
        dt = CFL * grids[0].dx
        res = integrate_serial(lr0s, grids, rngs, T_F, dt,
                               tol_sing_val=tol, drop_tol=tol, progress=False)
        best = min(best, res.runtime / (len(res.time) - 1))
    return best * 1e3


if __name__ == "__main__":
    Nxs = np.array([128, 256, 512, 1024, 2048])
    Nmus = np.array([64, 128, 256, 512])

    t_Nx_1 = np.array([time_per_step(Nx) for Nx in Nxs])
    t_Nx_2 = np.array([time_per_step(Nx, n_domains=2) for Nx in Nxs])
    t_Nmu = np.array([time_per_step(256, Nmu=Nmu) for Nmu in Nmus])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.loglog(Nxs, t_Nx_1, "o-", label="1 domain")
    ax1.loglog(Nxs, t_Nx_2, "s-", label="2 domains (serial)")
    ax1.loglog(Nxs, t_Nx_1[0] * Nxs / Nxs[0], "k--", label=r"$\mathcal{O}(N_x)$")
    ax1.set_xlabel("$N_x$")
    ax1.set_ylabel("time per step [ms]")
    ax1.legend()

    ax2.loglog(Nmus, t_Nmu, "o-", label=r"1 domain, $N_x=256$")
    ax2.loglog(Nmus, t_Nmu[0] * Nmus / Nmus[0], "k--", label=r"$\mathcal{O}(N_\mu)$")
    ax2.set_xlabel(r"$N_\mu$")
    ax2.legend()

    savepath = Path(__file__).resolve().parents[3] / "plots"
    savepath.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(savepath / "scaling.png", dpi=200)
    np.savez(savepath / "scaling.npz", Nxs=Nxs, Nmus=Nmus,
             t_Nx_1=t_Nx_1, t_Nx_2=t_Nx_2, t_Nmu=t_Nmu)
