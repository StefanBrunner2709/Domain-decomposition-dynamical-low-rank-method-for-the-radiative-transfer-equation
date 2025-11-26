# Domain-decomposition-dynamical-low-rank-method-for-the-radiative-transfer-equation
Dynamical low rank solver for the 1x1 dimensional and 2x1 dimensional radiative transfer equation, with and without domain decomposition. Can be used to reproduce the numerical experiments in the paper "Domain decomposition dynamical low-rank method for the radiative transfer equation" by Brunner, S., Einkemmer, L. and Haut, T.

## Installation
Install with `pip` or `uv` by
```shell
pip install .
```
or
```shell
uv pip install .
```

## Run results for publication
To reproduce all the results, run 
```shell
python3 DLR_rt/main.py
```
One can also just run a single example (possible options are "lattice", "hohlraum" or "pointsource")
```shell
python3 DLR_rt/main.py lattice
```
or multiple chosen examples
```shell
python3 DLR_rt/main.py hohlraum pointsource
```
In both the lattice and hohlraum example, first a reference solution on a single domain is calculated (with high rank), then a simulation on a single domain is run (with low rank) and then a simulation with domain decomposition is run.
For the pointsource example, two different simulations with domain decomposition are run.
