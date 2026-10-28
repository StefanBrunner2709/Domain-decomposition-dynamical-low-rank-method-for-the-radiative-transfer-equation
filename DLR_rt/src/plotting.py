"""
Shared plots for the 1x1d domain decomposition, saved as PNG in plots/.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.cm import ScalarMappable
from matplotlib.colors import BoundaryNorm, Normalize
from matplotlib.ticker import MaxNLocator

from DLR_rt.src.dd_1x1d import RunResult

PLOTS_DIR = Path(__file__).resolve().parents[2] / "plots"
FLOOR = 1e-17  # errors of exactly 0 are drawn at this value on log scales
# Fixed colour per phase, so a phase has the same colour in every plot
PHASES = ["PSI", "add_basis", "drop_basis", "F_b", "exchange"]


def save_fig(fig, name: str, dpi: int = 200) -> Path:
    """
    Save as plots/<name>.png and close the figure.
    """
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    path = PLOTS_DIR / f"{name}.png"
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return path


def interfaces(res: RunResult) -> list[float]:
    """
    x positions of the interfaces between neighbouring subdomains.
    """
    return [0.5 * (a.X[-1] + b.X[0]) for a, b in zip(res.grids, res.grids[1:])]


def rank_blocks(res: RunResult) -> list[int]:
    """
    Domain indices where the block of a new MPI rank starts (without 0).
    """
    return [i for i in range(1, len(res.owner)) if res.owner[i] != res.owner[i - 1]]


def plot_solution(ax, res: RunResult, f=None, title: str | None = None):
    """
    Plots f over (x, mu) with dashed lines at the interfaces, default f at t_f.
    """
    f = res.f() if f is None else f
    g0, g1 = res.grids[0], res.grids[-1]
    im = ax.imshow(
        f.T, extent=[g0.X[0], g1.X[-1], g0.MU[0], g0.MU[-1]],
        origin="lower", aspect="auto"
    )
    for x in interfaces(res):
        ax.axvline(x, color="w", ls="--", lw=0.8, alpha=0.6)
    ax.set(xlabel="$x$", ylabel=r"$\mu$",
           title=title or f"$f$ at $t={res.time[-1]:.3f}$")
    ax.figure.colorbar(im, ax=ax)
    return im


def plot_rank_history(ax, res: RunResult, max_legend: int = 8):
    """
    Rank of every subdomain over time, like plot (a) in Stefan's paper, Figure 6.
    """
    n = res.n_domains
    if n <= max_legend:
        colors = [f"C{i}" for i in range(n)]
    else:
        # Too many domains for a legend, we colour by position instead
        cmap, norm = plt.cm.viridis, Normalize(0, n - 1)
        colors = cmap(norm(np.arange(n)))

    for i, (hist, c) in enumerate(zip(res.rank_hist, colors)):
        ax.step(res.time, hist, where="post", color=c, lw=1.2, label=f"domain {i}")
    ax.set(xlabel="$t$", ylabel="rank $r(t)$", xlim=(res.time[0], res.time[-1]),
           title="Rank per subdomain")
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    if n <= max_legend:
        ax.legend(fontsize=8)
    else:
        ax.figure.colorbar(ScalarMappable(norm, cmap), ax=ax, label="subdomain")


def rank_matrix(res: RunResult, times) -> tuple[np.ndarray, np.ndarray]:
    """
    Ranks at the time steps closest to times, shape [n_times, n_domains].
    """
    time = np.asarray(res.time)
    idx = [int(np.argmin(np.abs(time - t))) for t in times]
    return time[idx], np.array(res.rank_hist)[:, idx].T


def plot_rank_snapshots(ax, res: RunResult, times, annotate: bool | None = None):
    """
    Rank of every subdomain at several times, like plot (b) in Stefan's paper.

    x: subdomain, y: time. Vertical lines separate the blocks of the MPI ranks,
    so the load of each rank can be read off directly.
    """
    t_snap, R = rank_matrix(res, times)
    n_t, n = R.shape
    # One colour per integer rank
    bounds = np.arange(R.min(), R.max() + 2) - 0.5
    norm = BoundaryNorm(bounds, plt.cm.viridis.N)
    im = ax.imshow(R, origin="lower", aspect="auto", cmap="viridis", norm=norm)
    if annotate is None:
        annotate = n <= 24 and n_t <= 16
    if annotate:
        # Dark text on the light end of viridis, white on the dark end
        mid = 0.5 * (R.min() + R.max())
        for (k, i), r in np.ndenumerate(R):
            ax.text(i, k, str(r), ha="center", va="center", fontsize=8,
                    color="k" if r > mid else "w")
    for i in rank_blocks(res):
        ax.axvline(i - 0.5, color="w", lw=2)
    if res.owner:
        # MPI ranks on top, at the centre of their block
        starts = [0] + rank_blocks(res)
        stops = rank_blocks(res) + [n]
        top = ax.secondary_xaxis("top")
        top.set_xticks([(a + b - 1) / 2 for a, b in zip(starts, stops)],
                       [f"P{p}" for p in range(len(starts))])
        top.tick_params(length=0)
    ax.set_xticks(range(n))
    ax.set_yticks(range(n_t), [f"{t:.2f}" for t in t_snap])
    ax.set(xlabel="subdomain", ylabel="$t$")
    cbar = ax.figure.colorbar(im, ax=ax, label="rank")
    cbar.set_ticks(np.arange(R.min(), R.max() + 1))


def plot_timings_per_rank(ax, res: RunResult):
    """
    Measured time per phase, stacked per MPI rank.
    """
    per_rank = res.timings_per_rank()
    x = np.arange(len(per_rank))
    bottom = np.zeros(len(per_rank))
    for phase in PHASES:
        values = np.array([tim.get(phase, 0.0) for tim in per_rank])
        if not values.any():
            continue
        ax.bar(x, values, bottom=bottom, width=0.6, label=phase,
               color=f"C{PHASES.index(phase)}", edgecolor="w", linewidth=1)
        bottom += values
    ax.set_xticks(x, [f"P{p}" for p in x])
    ax.set(xlabel="MPI rank", ylabel="time [s]", title="Measured time per rank")
    ax.legend(fontsize=8)


def plot_error_over_time(ax, t, curves: dict, floor: float = FLOOR):
    """
    Several error curves on a log scale, exact zeros are drawn at floor.
    """
    for label, err in curves.items():
        ax.semilogy(t, np.maximum(err, floor), label=label)
    if any(np.any(np.asarray(err) == 0) for err in curves.values()):
        ax.axhline(floor, color="k", lw=0.5, ls=":")
        ax.text(t[-1], floor, "0 (bit-identical)", ha="right", va="bottom", fontsize=8)
    ax.set(xlabel="$t$", ylabel=r"relative $L^2$ error")
    ax.legend(fontsize=8)


if __name__ == "__main__":
    # Demo with the serial version, the MPI blocks are only drawn for MPI results
    from DLR_rt.src.dd_1x1d import integrate_serial, setup_domains

    grids, lr0s, rngs = setup_domains(Nx=256, Nmu=64, r=5, n_domains=8, seed=0)
    res = integrate_serial(lr0s, grids, rngs, t_f=0.5, dt=1e-3,
                           tol_sing_val=1e-5, drop_tol=1e-5)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    plot_solution(axes[0], res)
    plot_rank_history(axes[1], res)
    plot_rank_snapshots(axes[2], res, times=np.linspace(0, 0.5, 6))
    print(f"Saved {save_fig(fig, 'demo_plotting')}")
