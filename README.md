# DLR-rt
Dynamical low rank solver for the 1x1 dimensional and 2x1 dimensional radiative transfer equation.

## Installation
Install with `pip` or `uv` by typing
```shell
pip install .
```
or
```shell
uv pip install .
```

## Run results for publication
To reproduce the results in "Domain decomposition with dynamical low rank methods for the 2x1 dimensional radiative transfer equation", run 
```shell
python3 DLR_rt/examples/dlr_2x1d_periodic_spacedepcoeff.py
```
and
```shell
python3 DLR_rt/examples/dlr_2x1d_dd_lattice_splitting.py
```
and
```shell
python3 DLR_rt/examples/dlr_2x1d_dd_hohlraum_splitting.py
```
