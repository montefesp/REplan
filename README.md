# py_grid_exp

This repository will contain Python code to automatically perform grid expansion planning.

The repository is organized as follows:

1. Code is contained in the src/ folder
2. Data is contained in the data/ folder available at [DoX](https://dox.ulg.ac.be/index.php/apps/files/?dir=/py_grid_exp&fileid=268947668)
3. Results are generated into the output/ folder.
	
## 1. The 'src/' folder

- network.py: Contains the class Network modelizing the different components of a network
- optimizer.py: Takes a Network object, transform it into an optimization model and solves it using a solver
- component_attrs/: Files describing the components of a network
- analyze/: Tools to visualize results
	- run_network_visual.py: Interactive visualization
- data/: Tools to manipulate data
- examples/: Main scripts making use of the data/ tools, network.py and optimize.py to built a special case of network and optimize it
	
## 2. The 'data/' folder

Nothing special to note about this directory except that all data is stocked there.

Also, the name of the folder should correspond to the names of the folder contained in 'src/data'. Then, the code in 'src/data/x' will mainly be used to 
manipulate data in 'data/x'.

## 3. The 'output/' folder

The 'output/' folder is not contained in the git but can be created with the two following subdirectories:

- examples/: This folder contains the results of the runs of the examples scripts.
- geographics/: Stores geographical shape allowing to not generate them at each run in the examples


