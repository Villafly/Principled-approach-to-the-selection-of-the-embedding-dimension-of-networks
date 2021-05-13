import numpy as np
import pandas as pd
from scipy import optimize
import argparse
import os
import cPickle as pickle
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from embedding_dist import cal_embedding_distance

#temp
import networkx as nx

def parse_args(graph_name):
    '''
    Parses the node2vec arguments.
    '''
    parser = argparse.ArgumentParser(description="Run node2vec.")
    
#    parser.add_argument('--input', nargs='?', default='./pic/conect_data/graphs/{}'.format(graph_name),
#                        help='Input graph path')
                        
    parser.add_argument('--input', nargs='?', default='./{}'.format(graph_name),
                        help='Input graph path')
    
    parser.add_argument('--output', nargs='?', default='./emb',
                        help='Embeddings path')
                                           
    parser.add_argument('--start_dim', type=int, default= 2,
                        help='the start embedding dimension. Default is 2.')
                                           
    parser.add_argument('--end_dim', type=int, default= 100,
                        help='the end embedding dimension. Default is 150.')
                                           
    parser.add_argument('--step', type=int, default= 4,
                        help='the step dimension from start_dim to end_dim. Default is 2.')                                        
     
    parser.add_argument('--length', type=int, default= 10,
                        help='Length of walk per source. Default is 10.')    
 
    parser.add_argument('--num-walks', type=int, default=20,
                        help='Number of walks per source. Default is 20.')
    
    parser.add_argument('--window_size', type=int, default=5,
                      	help='Context size for optimization. Default is 5.')
    
    parser.add_argument('--iter', default=10, type=int,
                        help='Number of epochs in SGD')
    
    parser.add_argument('--workers', type=int, default=8,
                        help='Number of parallel workers. Default is 8.')
    
    parser.add_argument('--p', type=float, default=1,
                        help='Return hyperparameter. Default is 1.')
    
    parser.add_argument('--q', type=float, default=1,
                        help='Inout hyperparameter. Default is 1.')
    
    parser.add_argument('--weighted', dest='weighted', action='store_true',
                        help='Boolean specifying (un)weighted. Default is unweighted.')
                                           
    parser.add_argument('--unweighted', dest='unweighted', action='store_false')
    parser.set_defaults(weighted=False)
    
    parser.add_argument('--directed', dest='directed', action='store_true',
                        help='Graph is (un)directed. Default is undirected.')
                                           
    parser.add_argument('--undirected', dest='undirected', action='store_false')
    parser.set_defaults(directed=False)

    return parser.parse_args()
 

   

if __name__ == "__main__":
 
      args = parse_args('football')
      norm_loss = cal_embedding_distance(args)    
