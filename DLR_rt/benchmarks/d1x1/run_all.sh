#!/usr/bin/env bash
# Rerun all 1x1d benchmarks of the thesis, one after the other.
#
#   DLR_rt/benchmarks/d1x1/run_all.sh                    everything (several hours)
#   DLR_rt/benchmarks/d1x1/run_all.sh scaling balancing  only these groups
#   DLR_rt/benchmarks/d1x1/run_all.sh -n                 only print the commands
#   nohup DLR_rt/benchmarks/d1x1/run_all.sh > run_all.out 2>&1 &   (survives logout)
#
# Groups: serial correctness scaling balancing presentation
#
# Before the first run, the old results are moved (not deleted, plots/ and data/ are
# not in git) to plots/archive/<date>/ and data/benchmarks/archive/<date>/.
# The .npy/.pkl files for replot_presentation stay in data/benchmarks/.
# Every benchmark writes its output to data/benchmarks/logs/<name>.log. A failing
# benchmark does not stop the others, the summary at the end lists it.
#
# Clean timings: nothing else running on the machine, the MPI ranks are bound to
# cores 0..15, the serial runs are pinned to core $SERIAL_CORE.

set -u

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
ENV_BIN="${ENV_BIN:-$HOME/.conda/envs/domain/bin}"
PY="$ENV_BIN/python"
MPIRUN="$ENV_BIN/mpirun --bind-to core --map-by core"
NP="${NP:-16}"                  # max MPI processes
SERIAL_CORE="${SERIAL_CORE:-20}"
STAMP="$(date +%Y-%m-%d_%H%M)"
LOGS="$ROOT/data/benchmarks/logs"
B="DLR_rt.benchmarks.d1x1"
STD="--Nx 2048 --Nmu 128 --tol 1e-7 --repeat 3"   # the standard setup of the thesis

DRY=0
SELECTED=()
for arg in "$@"; do
    case "$arg" in
        -n|--dry-run) DRY=1 ;;
        serial|correctness|scaling|balancing|presentation) SELECTED+=("$arg") ;;
        *) echo "Unknown argument: $arg"; exit 1 ;;
    esac
done
[ ${#SELECTED[@]} -eq 0 ] && SELECTED=(serial correctness scaling balancing presentation)

FAILED=()
T_START=$(date +%s)

# run <name> <command...>: log to $LOGS/<name>.log, continue on failure
run() {
    local name="$1"; shift
    echo "[$(date +%H:%M:%S)] $name: $*"
    [ "$DRY" -eq 1 ] && return
    local t0=$(date +%s)
    if (cd "$ROOT" && "$@") > "$LOGS/$name.log" 2>&1; then
        echo "           done in $(( ($(date +%s) - t0) / 60 )) min"
    else
        echo "           FAILED after $(( ($(date +%s) - t0) / 60 )) min, see $LOGS/$name.log"
        FAILED+=("$name")
    fi
}

### Checks before anything is changed
if [ "$DRY" -eq 0 ]; then
    # python -m DLR_rt.benchmarks..., the old benchmark scripts or a running mpirun
    BUSY='-m DLR_rt\.benchmarks|benchmark_mpi_scaling|prterun|mpirun'
    if pgrep -f -- "$BUSY" | grep -vx "$$" > /dev/null; then
        echo "Another benchmark is running, stop it first:"
        pgrep -af -- "$BUSY" | grep -v "^$$ "
        exit 1
    fi
    load=$(cut -d' ' -f1 /proc/loadavg)
    echo "Load average: $load (should be well below 1 for clean timings)"
    if [ -n "$(cd "$ROOT" && git status --porcelain --untracked-files=no)" ]; then
        echo "WARNING: uncommitted changes, the records will show git_dirty = true"
    fi

    ### Archive the old results
    mkdir -p "$ROOT/plots/archive/$STAMP" "$ROOT/data/benchmarks/archive/$STAMP" "$LOGS"
    find "$ROOT/plots" -mindepth 1 -maxdepth 1 ! -name archive ! -name .gitkeep \
        -exec mv -t "$ROOT/plots/archive/$STAMP" {} +
    find "$ROOT/data/benchmarks" -maxdepth 1 -name "*.json" \
        -exec mv -t "$ROOT/data/benchmarks/archive/$STAMP" {} +
    find "$LOGS" -maxdepth 1 -name "*.log" \
        -exec mv -t "$ROOT/data/benchmarks/archive/$STAMP" {} +
    echo "Old plots    -> plots/archive/$STAMP"
    echo "Old results  -> data/benchmarks/archive/$STAMP"
fi
echo "Groups: ${SELECTED[*]}"
echo

for group in "${SELECTED[@]}"; do
    case "$group" in

    serial)
        for e in nx nmu tol domains memory; do
            run "serial_$e" taskset -c "$SERIAL_CORE" $PY -m $B.serial $e
        done
        run serial_combined $PY -m $B.serial plot combined
        ;;

    correctness)
        run correctness $MPIRUN -n 8 $PY -m $B.correctness
        run correctness_compare $MPIRUN -n 4 $PY -m $B.correctness compare 8
        ;;

    scaling)
        # Strong scaling, 16 subdomains, every inflow
        for inflow in tanh beam pulse two_beam sharp_beam; do
            run "strong_${inflow}_Nx2048" $MPIRUN -n $NP $PY -m $B.scaling strong \
                $STD --t_f 1.0 --inflow $inflow
        done
        run strong_compare_Nx2048 $PY -m $B.scaling plot compare --Nx 2048
        # Larger and smaller problem, fewer subdomains
        run strong_tanh_Nx4096 $MPIRUN -n $NP $PY -m $B.scaling strong \
            --Nx 4096 --Nmu 128 --tol 1e-7 --repeat 3 --t_f 0.5 --inflow tanh
        run strong_tanh_Nx128 $MPIRUN -n $NP $PY -m $B.scaling strong \
            --Nx 128 --Nmu 128 --tol 1e-7 --repeat 3 --t_f 1.0 --inflow tanh
        run strong_tanh_Nx2048_n8_tf05 $MPIRUN -n 8 $PY -m $B.scaling strong \
            $STD --t_f 0.5 --inflow tanh --n_domains 8 --suffix _n8_tf05
        # Weak scaling, 512 grid points per process
        run weak_tanh_Nx512 $MPIRUN -n $NP $PY -m $B.scaling weak \
            --Nx 512 --Nmu 128 --tol 1e-7 --repeat 3 --t_f 0.5 --inflow tanh
        # More subdomains per process, 8 processes
        run domains_tanh_Nx2048 $MPIRUN -n 8 $PY -m $B.scaling domains \
            $STD --t_f 0.5 --inflow tanh --ranks 8
        # Smaller subdomains at the boundaries or in the middle
        run asymmetric_tanh_Nx2048 $MPIRUN -n 8 $PY -m $B.scaling asymmetric \
            $STD --t_f 0.5 --inflow tanh --ranks 8
        run asymmetric_beam_Nx2048 $MPIRUN -n 8 $PY -m $B.scaling asymmetric \
            $STD --t_f 1.0 --inflow beam --ranks 8
        ;;

    balancing)
        # Static: even vs. balanced split with measured weights
        run balanced_tanh_Nx2048_n32 $MPIRUN -n 8 $PY -m $B.balancing balanced \
            $STD --t_f 0.5 --inflow tanh --n_domains 32
        run balanced_tanh_Nx512_n16 $MPIRUN -n 8 $PY -m $B.balancing balanced \
            --n_domains 16
        # Dynamic: tol 1e-8 for higher ranks, t_f 1.5 while the structure moves,
        # 2 resp. 4 subdomains per process at p = 16
        run dynamic_tanh_Nx2048_n32 $MPIRUN -n $NP $PY -m $B.balancing dynamic \
            --Nx 2048 --Nmu 64 --tol 1e-8 --repeat 3 --t_f 1.5 --inflow tanh \
            --n_domains 32 --ranks 1 8 16 --every 25 100 --suffix _tf15_tol8
        run dynamic_beam_Nx2048_n64 $MPIRUN -n $NP $PY -m $B.balancing dynamic \
            --Nx 2048 --Nmu 64 --tol 1e-8 --repeat 3 --t_f 1.5 --inflow beam \
            --n_domains 64 --ranks 1 8 16 --every 25 100 --min_gain 0.10 \
            --suffix _tf15_tol8
        ;;

    presentation)
        run replot_presentation $PY -m DLR_rt.examples.d1x1.replot_presentation
        ;;
    esac
done

echo
echo "Total: $(( ($(date +%s) - T_START) / 60 )) min"
if [ ${#FAILED[@]} -gt 0 ]; then
    echo "FAILED: ${FAILED[*]}"
    exit 1
fi
if [ "$DRY" -eq 0 ]; then
    echo "All benchmarks finished, results in data/benchmarks/, plots in plots/"
fi
