"""
Runs selected examples from publication.
Usage:
    python3 main.py                         # runs all
    python3 main.py lattice                 # run only lattice
    python3 main.py hohlraum pointsource    # run multiple
Possible examples: lattice, hohlraum, pointsource
"""

import sys

from DLR_rt.examples.publication.dlr_2x1d_dd_hohlraum_splitting import run_dd_hohlraum
from DLR_rt.examples.publication.dlr_2x1d_dd_lattice_splitting import run_dd_lattice
from DLR_rt.examples.publication.dlr_2x1d_periodic_spacedepcoeff import run_1d

VALID = {
    "lattice",
    "hohlraum",
    "pointsource"
}

if __name__ == "__main__":

    # If no arguments → run all examples
    if len(sys.argv) == 1:
        to_run = VALID
    else:
        to_run = set(sys.argv[1:])       # all arguments after main.py
        unknown = to_run - VALID
        if unknown:
            print("Error: unknown example(s):", ", ".join(unknown))
            print("Valid options:", ", ".join(VALID))
            sys.exit(1)

    # ---- LATTICE EXAMPLE ----
    if "lattice" in to_run:
        ### Run lattice example
        # Generate reference solution on 1 domain
        print("Running lattice reference solution on 1 domain...")
        run_1d(option_problem="lattice", option_calculate_ref=True)

        # Simulation on a single domain
        print("Running lattice simulation on 1 domain...")
        run_1d(option_problem="lattice", option_error_estimate=True)

        # Simulation with domain decomposition
        print("Running lattice simulation with domain decomposition...")
        run_dd_lattice(option_error_estimate=True, option_dof_plot=True)

    # ---- HOHLRAUM EXAMPLE ----
    if "hohlraum" in to_run:
        ### Run hohlraum example
        # Generate reference solution on 1 domain
        print("Running hohlraum reference solution on 1 domain...")
        run_1d(option_problem="hohlraum", option_calculate_ref=True)

        # Simulation on a single domain
        print("Running hohlraum simulation on 1 domain...")
        run_1d(option_problem="hohlraum", option_error_estimate=True)

        # Simulation with domain decomposition
        print("Running hohlraum simulation with domain decomposition...")
        run_dd_hohlraum(option_problem="hohlraum", option_error_estimate=True, 
                        option_dof_plot=True)

    # ---- POINTSOURCE EXAMPLE ----
    if "pointsource" in to_run:
        ### Run isotropic pointsource example
        # Simulation with larger tolerance
        print("Running pointsource simulation with domain " + 
              "decomposition and large tolerance...")
        run_dd_hohlraum(option_problem="pointsource")

        # Simulation with smaller tolerance
        print("Running pointsource simulation with domain " + 
              "decomposition and small tolerance...")
        run_dd_hohlraum(option_problem="pointsource_2")
