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

- <a href="https://github.com/PyPSA/PyPSA">PyPSA</a> >= 0.17.0
- <a href="https://github.com/montefesp/iepy"iepy</a>
- <a href="https://github.com/montefesp/resite"resite</a>
