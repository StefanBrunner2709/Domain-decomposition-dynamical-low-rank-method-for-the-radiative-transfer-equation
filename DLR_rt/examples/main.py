"""
Runs examples from publication.
"""

from DLR_rt.examples.publication.dlr_2x1d_dd_hohlraum_splitting import run_dd_hohlraum
from DLR_rt.examples.publication.dlr_2x1d_dd_lattice_splitting import run_dd_lattice
from DLR_rt.examples.publication.dlr_2x1d_periodic_spacedepcoeff import run_1d

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




### Run isotropic pointsource example
# Simulation with larger tolerance
print("Running pointsource simulation with domain decomposition and large tolerance...")
run_dd_hohlraum(option_problem="pointsource")

# Simulation with smaller tolerance
print("Running pointsource simulation with domain decomposition and small tolerance...")
run_dd_hohlraum(option_problem="pointsource_2")
