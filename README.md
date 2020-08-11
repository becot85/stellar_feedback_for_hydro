# Stellar feedback for hydro

Requirements to use this repository

* [NuPyCEE](https://github.com/NuGrid/NuPyCEE) (chemical enrichment feedback)
* [FSPS](https://github.com/cconroy20/fsps) (radiative feedback)
* ...

## Installation instructions

#### NuPyCEE
* Clone NuPyCEE (https://github.com/NuGrid/NuPyCEE.git) and install with `python setup.py install`.

#### pyFSPS
* Install FSPS (https://github.com/cconroy20/fsps) following the instructions in `doc/INSTALL`
* Clone the pyFSPS wrapper (https://github.com/dfm/python-fsps.git). Install it with `python setup.py install`; if it fails, you may need recompile FSPS but add the `-fPIC` flag to `F90FLAGS` in `src/Makefile`