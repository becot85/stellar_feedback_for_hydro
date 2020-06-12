# Stellar feedback for hydro

Requirements to use this repository

* [NuPyCEE](https://github.com/NuGrid/NuPyCEE) (chemical enrichment feedback)
* [FSPS](https://github.com/cconroy20/fsps) (radiative feedback)
* ...

## Installation instructions

#### NuPyCEE
* From within the *stellar\_feedback\_for\_hydro* directory ..
* git clone https://github.com/NuGrid/NuPyCEE.git

#### pyFSPS
* Clone andn build the FSPS Fortran code (may need -fPIC flag even when not using Intel compilers)
* Clone and build the pyFSPS wrapper