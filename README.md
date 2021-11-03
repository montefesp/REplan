<a href="https://www.montefiore.uliege.be/"><img src="https://www.montefiore.uliege.be/upload/docs/image/svg-xml/2019-04/montefiore_institute.svg" alt="University of Liège - Montefiore institute" width="230px"></a>


# REplan

REplan provides a set of methods enabling the set-up of spatially-resolved expansion planning problems.
Currently, the tool is under development and focuses on electricity transmission grids at a macro-level in Europe 
(i.e. our most granular model uses NUTS3 adminisrative regions). This particular release of the repository is linked to
Chapter 9 ("On the role of complementarity in siting renewable power generation assets and its economic implications 
for power systems") of the following publication: "Complementarity of Variable Renewable Energy Sources", 
ISBN: 9780323855273, available [here](https://www.elsevier.com/books/complementarity-of-variable-renewable-energy-sources/jurasz/978-0-323-85527-3). 

The repository is organized as follows:

1. network/: Various building blocks for setting up a power network via PyPSA objects
2. projects/: Location of the main file used to run the script
3. postprocessing/: Tools for analysing the results generated via the projects (not up to date)
4. tests/: Various unit tests

## Installation process

In order to set-up the `REplan` repository, the following steps are required:

- set-up [this](https://github.com/dcradu/resite_ip/releases/edit/v0.0.2) release of `resiteIP` according to the instructions in the `readme` file
- clone the [EPIPPy](https://github.com/montefesp/EPIPPy) repository (a module that helps pre-processing input data and providing it in proper formats)
- create a `Python` environment from the `environment.yaml` file

## Example run

Once all dependencies are installed, a typical run is done via the following procedure:

- run `resiteIP` ex-ante in order to obtain the optimal deployment of RES sites in a `pickle` format
- configure the target system via the `config.yaml` file in `projects/book/` (this step also includes providing paths to the results of `resiteIP`)
- run the model via the `main.py` file in the same folder

Results of the expansion planning problem are stored in a typical `PyPSA` format relying on `.csv` files. Note that a valid license for `gurobi` or `cplex` are required in order to solve the resulting instances.

# License
This software is released as free software under the [GPLv3](http://www.gnu.org/licenses/gpl-3.0.en.html).
