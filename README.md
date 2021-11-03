<a href="https://www.montefiore.uliege.be/"><img src="https://www.montefiore.uliege.be/upload/docs/image/svg-xml/2019-04/montefiore_institute.svg" alt="University of Liège - Montefiore institute" width="230px"></a>


# REplan

REplan provides a set of methods enabling the set-up of spatially-resolved expansion planning problems.
Currently, the tool is under development and focuses on electricity transmission grids at a macro-level in Europe (i.e. our most granular model uses NUTS3 adminisrative regions).

The repository is organized as follows:

1. network/: Various building blocks for setting up a power network via PyPSA objects
2. projects/: Location of the main file used to run the script
3. postprocessing/: Tools for analysing the results generated via the projects (not up to date)
4. tests/: Various unit tests

## Dependencies

The main dependencies of this package are:

- [PyPSA](https://github.com/PyPSA/PyPSA) (at least v0.17, as the modelling framework enabling the development of expansion planning problems)
- [EPIPPy](https://github.com/montefesp/EPIPPy) (as a module that helps pre-processing input data and providing it in proper formats)
- [resiteIP](https://github.com/dcradu/resite_ip/tree/feature-book) (as the module that enables the siting of RES assets)

In addition to these three modules, some other minor dependencies are provided via a `.yaml` file.

## Example run

Once all dependencies are installed, a typical run is done via the following procedure:

- run `resiteIP` ex-ante in order to obtain the optimal deployment of RES sites
- configure the target system via the `config.yaml` file in `projects/book/` (this step also includes providing paths to the results of `resiteIP`)
- run the model via the `main.py` file in the same folder

Results of the expansion planning problem are stored in a typical `PyPSA` format relying on `.csv` files. Note that a valid license for `gurobi` or `cplex` are required in order to solve the resulting instances.

# License
This software is released as free software under the [GPLv3](http://www.gnu.org/licenses/gpl-3.0.en.html).
