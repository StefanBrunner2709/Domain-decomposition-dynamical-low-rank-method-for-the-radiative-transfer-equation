"""
All plots of the presentation (HPC_Aspects_of_RT) with larger fonts, from saved results.

No simulation is run. The data are the ones the slides were made with: the result of
the 8-domain run (pkl, npy) and the JSONs archived on 2026-11-17 before the benchmarks
were rerun with the faster code, so the numbers stay the same as on the slides.
Output: plots/presentation/, same file names as the slide images. Backup II (asymmetric
splitting, 4 panels) is missing: the result of that run was not saved.

    slide  file
    7      tanh_Nx2048_solution.png
    8      serial_combined.png
    9-11   bench_domains_tanh_Nx2048_8_{solution,ranks,timings}.png
    B III  correctness.png
    B IV   bench_strong_tanh_Nx2048_n8_tf05.png

    python -m DLR_rt.examples.d1x1.replot_presentation [--scale 1.0]

--scale multiplies all font sizes once more (1.0: the sizes in FONT_SCALE below).
"""
from DLR_rt.benchmarks.d1x1 import common as C  # first, sets the BLAS threads

import argparse  # noqa: E402
import io  # noqa: E402
import json  # noqa: E402
import pickle  # noqa: E402
from pathlib import Path  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.text import Text  # noqa: E402

from DLR_rt.benchmarks.d1x1 import correctness, scaling, serial  # noqa: E402
import DLR_rt.src.d1x1.plotting as P  # noqa: E402
from DLR_rt.src.d1x1.domains import RunResult, setup_domains  # noqa: E402
from DLR_rt.src.d1x1.lowrank import LR  # noqa: E402

DATA_DIR = C.DATA_DIR
ARCHIVE = DATA_DIR / "archive" / "2026-11-17_1738"  # JSONs of the slides
OUT_DIR = P.PLOTS_DIR / "presentation"

# Large enough to read from the back of the room (single-panel figures)
STYLE = {"font.size": 18, "axes.titlesize": 24, "axes.labelsize": 24,
         "xtick.labelsize": 18, "ytick.labelsize": 18, "legend.fontsize": 18}

# Factor on all fonts of a figure, larger for the figures that are small on the slide
# or were made with the small benchmark fonts. 1.0 gives the images in the slides.
FONT_SCALE = {
    "tanh_Nx2048_solution": 1.45,
    "serial_combined": 1.45,
    "bench_domains_tanh_Nx2048_8_solution": 1.2,
    "bench_domains_tanh_Nx2048_8_ranks": 1.2,
    "bench_domains_tanh_Nx2048_8_timings": 1.2,
    "correctness": 1.4,
    "bench_strong_tanh_Nx2048_n8_tf05": 1.6,
}


class _Unpickler(pickle.Unpickler):
    """Results pickled before the restructure: the old classes -> the d1x1 ones."""
    MOVED = {("DLR_rt.src.dd_1x1d", "RunResult"): RunResult,
             ("DLR_rt.src.lr", "LR"): LR}

    def find_class(self, module, name):
        return self.MOVED.get((module, name)) or super().find_class(module, name)


def load_result(name: str = "domains_tanh_Nx2048_8") -> RunResult:
    return _Unpickler(io.BytesIO((DATA_DIR / f"{name}_result.pkl").read_bytes())).load()


def load_archived(name: str) -> dict:
    return json.loads((ARCHIVE / f"{name}.json").read_text())


### Fonts

def scale_fonts(fig, k: float):
    """All texts of fig k times larger: titles, labels, ticks, legends, annotations."""
    if k == 1:
        return
    seen = set()
    for text in fig.findobj(Text):
        if id(text) not in seen:
            seen.add(id(text))
            text.set_fontsize(text.get_fontsize() * k)
    # Ticks created later (at drawing) copy these sizes, not the ones set above
    axes = list(fig.axes)
    for ax in fig.axes:
        axes += ax.child_axes  # e.g. the processes on top of the rank snapshots
    for ax in axes:
        for axis in (ax.xaxis, ax.yaxis):
            for which, ticks in (("major", axis.get_major_ticks()),
                                 ("minor", axis.get_minor_ticks())):
                if ticks:
                    axis.set_tick_params(which=which,
                                         labelsize=ticks[0].label1.get_fontsize())


def save(fig, name: str, k: float, out_dir: Path = OUT_DIR) -> Path:
    scale_fonts(fig, k)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{name}.png"
    # Larger texts need more room between the panels. Not with the legend outside
    # (timings), the axes would shrink to make room for it.
    if k != 1 and not name.endswith("_timings"):
        fig.tight_layout()
    # bbox "tight" also for the fixed-canvas benchmark figures, long titles would be
    # cut off otherwise
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return path


### Figures, one function each, all return (name, fig)

def solution_from_npy(name: str = "tanh_Nx2048", Nx: int = 2048, Nmu: int = 128,
                      t_f: float = 1.0):
    """f at t_f on one domain, the grid only gives the extent of the image."""
    f = np.load(DATA_DIR / f"{name}_f_final.npy")
    assert f.shape == (Nx, Nmu), f"Expected f of shape ({Nx}, {Nmu}), got {f.shape}"
    g = setup_domains(Nx, Nmu, 5, 1, 0)[0][0]
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(8, 6))
        im = ax.imshow(f.T, extent=[g.X[0], g.X[-1], g.MU[0], g.MU[-1]],
                       origin="lower", aspect="auto")
        ax.set(xlabel="$x$", ylabel=r"$\mu$", title=f"$f$ at $t={t_f}$")
        fig.colorbar(im, ax=ax)
    return f"{name}_solution", fig


def details_solution(res: RunResult, out: str):
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(8, 6))
        P.plot_solution(ax, res, title=f"$f$ at $t={res.time[-1]:.1f}$")
        for line in ax.lines:  # thicker interface lines, visible on a projector
            line.set(lw=1.5, alpha=0.8)
    return f"{out}_solution", fig


def details_ranks(res: RunResult, out: str):
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(8, 7))
        P.plot_rank_snapshots(ax, res, times=np.linspace(0, res.time[-1], 11))
        for text in ax.texts:  # numbers in the cells
            text.set_fontsize(14)
        ax.set_title("Low rank $r$ per subdomain", pad=12)
        cbar_ax = fig.axes[-1]  # every integer r overlaps with the larger fonts
        ticks = cbar_ax.get_yticks()
        cbar_ax.set_yticks(ticks[::int(np.ceil(len(ticks) / 10))])
    return f"{out}_ranks", fig


def details_timings(res: RunResult, out: str):
    with plt.rc_context(STYLE):
        # Narrower than it looks: the legend outside adds width, the saved image keeps
        # the aspect ratio of the earlier version (about 1.8 : 1)
        fig, ax = plt.subplots(figsize=(9.2, 6))
        P.plot_timings_per_rank(ax, res)
        ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1), frameon=False)
    return f"{out}_timings", fig


def serial_combined():
    return "serial_combined", serial.combined_figure(
        load_archived("serial_nx")["records"], load_archived("serial_memory")["records"])


def correctness_errors():
    content = load_archived("correctness")
    return "correctness", correctness.correctness_figure(
        content["results"], content.get("base", correctness.BASE))


def strong_scaling(name: str = "tanh_Nx2048_n8_tf05"):
    return f"bench_strong_{name}", scaling.summary_figure(
        "strong", load_archived(f"strong_{name}")["records"])


def all_figures():
    res = load_result()
    out = "bench_domains_tanh_Nx2048_8"
    yield solution_from_npy()
    yield serial_combined()
    yield details_solution(res, out)
    yield details_ranks(res, out)
    yield details_timings(res, out)
    yield correctness_errors()
    yield strong_scaling()


if __name__ == "__main__":
    ps = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ps.add_argument("--scale", type=float, default=1.0,
                    help="extra factor on all fonts, 1.0 = FONT_SCALE as it is")
    ps.add_argument("--out", type=Path, default=OUT_DIR)
    args = ps.parse_args()
    for name, fig in all_figures():
        k = FONT_SCALE[name] * args.scale
        print(f"Saved {save(fig, name, k, args.out)}  (fonts x{k:.2f})")
