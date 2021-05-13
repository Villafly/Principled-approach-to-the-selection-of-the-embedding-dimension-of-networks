# Principled-approach-to-the-selection-of-the-embedding-dimension-of-networks

# defining_identifying_optimal_dimension
This repository provides a reference implementation of the following paper:

### Basic Usage

#### Example
To run *defining_identifying_optimal_dimension* on the American college football network, execute the following command from the project home directory:<br/>
	``python2.7 define_identify.py --input graph/football``

#### Options
You can check out the other options available to use with *defining_identifying_optimal_dimension* using:<br/>
	``python2.7 define_identify.py --help``
  
#### Input
The supported input format is an edgelist:
	``node1_id_int node2_id_int <weight_float, optional>``		
The graph is assumed to be undirected and unweighted by default.

#### Output
The output is the optimal dimension identified by our algorithm.

