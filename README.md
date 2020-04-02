# py_ggrid

Py_GGrid provides a set of tools to ease the set up and simulation of grid expansion planning.
Currently, the tool is on development and focuses on electricity transmission grids at a macro-level (i.e. our most granular model uses as indivisible units NUTS3 regions).

We currently work using the PyPSA framework as network modelling tool.

The repository is organized as follows:

1. Code is contained in the src/ folder
2. Data is contained in the data/ folder and (regurlarly) uploaded at [DoX](https://dox.ulg.ac.be/index.php/apps/files/?dir=/py_grid_exp&fileid=268947668)
3. Results are generated into the output/ folder.
	
## 1. The 'src/' folder

The code is divided into a set of folders having different purposes:

- data/: Tools for manipulate data about energy systems (e.g. legacy capacity, potential capacity, load profiles, existing; topologies, capacity factors, ...), about geography and demographics (countries shapes, population, land use, ...) and about emission levels;
- network_builder: Tools for adding generators and storage units a PyPSA network;
- parameters/: Tools for manipulating techno-economic parameters stored in files in this folder;
- postprocessing/: Tools to visualize results;
- resite/: This folder corresponds to a sub-repository that serves as a pv and wind siting tool that can be used on its own;
- sizing/: This folder contains the scripts that make use of all previously mentionned tools to built a PyPSA network.

## 2. The 'data/' folder

Nothing special to note about this directory except that all data is stocked there.

Also, the name of the folder should correspond to the names of the folder contained in 'src/data'. Then, the code in 'src/data/x' will mainly be used to manipulate data in 'data/x'.

## 3. The 'output/' folder

The 'output/' folder is not contained in the git but can be created with the three following subdirectories:

- geographics/: Stores geographical shape allowing to not generate them at each run in the examples;
- resite/: This folder is destined at containing the output of the resite tool;
- sizing/: This folder contains the results of the runs of the examples scripts.

