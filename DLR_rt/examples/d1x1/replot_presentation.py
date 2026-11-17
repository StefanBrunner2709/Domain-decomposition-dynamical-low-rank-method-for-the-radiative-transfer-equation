"""
Single-panel plots with large fonts for the presentation, from saved results.

No simulation is run, only the saved data in data/benchmarks/ is plotted:
    tanh_Nx2048_f_final.npy          -> plots/tanh_Nx2048_solution.png
    domains_tanh_Nx2048_8_result.pkl -> plots/bench_domains_tanh_Nx2048_8_{solution,
                                        ranks,timings}.png

    python -m DLR_rt.examples.d1x1.replot_presentation
"""
import io
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import DLR_rt.src.d1x1.plotting as P
from DLR_rt.src.d1x1.domains import RunResult, setup_domains
from DLR_rt.src.d1x1.lowrank import LR

DATA_DIR = Path(__file__).resolve().parents[3] / "data" / "benchmarks"


class _Unpickler(pickle.Unpickler):
    """Results pickled before the restructure: the old classes -> the d1x1 ones."""
    MOVED = {("DLR_rt.src.dd_1x1d", "RunResult"): RunResult,
             ("DLR_rt.src.lr", "LR"): LR}

    def find_class(self, module, name):
        return self.MOVED.get((module, name)) or super().find_class(module, name)

# Large enough to read from the back of the room
STYLE = {"font.size": 18, "axes.titlesize": 24, "axes.labelsize": 24,
         "xtick.labelsize": 18, "ytick.labelsize": 18, "legend.fontsize": 18}


def solution_from_npy(name: str = "tanh_Nx2048", Nx: int = 2048, Nmu: int = 128,
                      t_f: float = 1.0):
    """f at t_f on one domain, the grid only gives the extent of the image."""
    f = np.load(DATA_DIR / f"{name}_f_final.npy")
    assert f.shape == (Nx, Nmu), f"Expected f of shape ({Nx}, {Nmu}), got {f.shape}"
    g = setup_domains(Nx, Nmu, 5, 1, 0)[0][0]
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(f.T, extent=[g.X[0], g.X[-1], g.MU[0], g.MU[-1]],
                   origin="lower", aspect="auto")
    ax.set(xlabel="$x$", ylabel=r"$\mu$", title=f"$f$ at $t={t_f}$")
    fig.colorbar(im, ax=ax)
    print(f"Saved {P.save_fig(fig, f'{name}_solution')}")


def details_from_pkl(name: str = "domains_tanh_Nx2048_8"):
    """Solution, low rank r per subdomain and time per process, one figure each."""
    res = _Unpickler(io.BytesIO((DATA_DIR / f"{name}_result.pkl").read_bytes())).load()
    out = f"bench_{name}"

    fig, ax = plt.subplots(figsize=(8, 6))
    P.plot_solution(ax, res, title=f"$f$ at $t={res.time[-1]:.1f}$")
    for line in ax.lines:  # thicker interface lines, visible on a projector
        line.set(lw=1.5, alpha=0.8)
    print(f"Saved {P.save_fig(fig, f'{out}_solution')}")

    fig, ax = plt.subplots(figsize=(8, 7))
    P.plot_rank_snapshots(ax, res, times=np.linspace(0, res.time[-1], 11))
    for text in ax.texts:  # numbers in the cells
        text.set_fontsize(14)
    ax.set_title("Low rank $r$ per subdomain", pad=12)
    print(f"Saved {P.save_fig(fig, f'{out}_ranks')}")

    # Narrower than it looks: the legend outside adds width, the saved image keeps
    # the aspect ratio of the earlier version (about 1.8 : 1)
    fig, ax = plt.subplots(figsize=(9.2, 6))
    P.plot_timings_per_rank(ax, res)
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1), frameon=False)
    print(f"Saved {P.save_fig(fig, f'{out}_timings')}")


if __name__ == "__main__":
    with plt.rc_context(STYLE):
        solution_from_npy()
        details_from_pkl()
