<a href="https://www.montefiore.uliege.be/"><img src="https://www.montefiore.uliege.be/upload/docs/image/svg-xml/2019-04/montefiore_institute.svg" alt="University of Liège - Montefiore institute" width="230px"></a>


# replan

REplan provides a set of tools to ease the set up and simulation of grid expansion planning.
Currently, the tool is on development and focuses on electricity transmission grids at a macro-level (i.e. our most granular model uses as indivisible units NUTS3 regions).

We currently work using the PyPSA framework as network modelling tool.

The repository is organized as follows:

1. network/: Tools for building a PyPSA network
2. projects/: Example scripts using the tools in network
3. postprocessing/: Tools for analysing the results generated via the projects
4. tests/: Unit tests

## Dependencies

- [PyPSA](https://github.com/PyPSA/PyPSA) >= 0.17.0
- [iepy](https://github.com/montefesp/iepy)
- [resite](https://github.com/montefesp/resite)

# License
This software is released as free software under the [GPLv3](http://www.gnu.org/licenses/gpl-3.0.en.html).
