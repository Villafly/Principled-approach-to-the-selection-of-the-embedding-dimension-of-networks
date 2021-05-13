import numpy as np
import networkx as nx
import node2vec
from gensim.models import Word2Vec
from scipy.spatial.distance import cdist
from scipy import optimize
import math
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def read_graph(args):
    '''
    Reads the input network in networkx.
    '''
    if args.weighted:
        G = nx.read_edgelist(args.input, nodetype=int, data=(('weight',float),), create_using=nx.DiGraph())
    else:
        G = nx.read_edgelist(args.input, nodetype=int, create_using=nx.DiGraph())
        for edge in G.edges():
            G[edge[0]][edge[1]]['weight'] = 1
    #G = nx.read_edgelist(file_name, nodetype=int)
    if not args.directed:
        G = G.to_undirected()
    return G
    
def fitting_func(dims,s,a,L):  
  return s/np.power(dims,a) + L
  
def identify_optimal_dim(embedding_dims, loss):
    '''
    Identify the optimal dimension range and compute the curve fitting parameter for graph.
    '''  
    (s,a,l),cov = optimize.curve_fit(fitting_func, embedding_dims,loss)
    fit_values = (fitting_func(np.array(embedding_dims),s,a,l))
    MSE = ((np.array(loss)-np.array(fit_values))**2).mean()
    opt = np.power((s/0.05),1/a)
    print 'the optimal dimension at 0.05 accuracy level is {}'.format(int(math.ceil(opt)))
    print 'the MSE of curve fitting is {}'.format(MSE)

def cal_cosine_matrices(G,walks,args):
    '''
    Compute the cosine distance between every node pair over different embedding dimensions.
    '''
    norm_loss = []
    walks = [map(str, walk) for walk in walks]
    node_num = len(G.nodes())
    if node_num < args.end_dim:
      args.end_dim = node_num 
    embedding_dims = range(args.start_dim,args.end_dim,args.step)
    if node_num < 500:
      embedding_dims.insert(0,node_num)
      print 'graph size smaller than the default end dimension, thus has been automatically set to {}'.format(node_num)
    else:
      embedding_dims.insert(0,500)  
    #cosine_matrices = np.zeros((len(embedding_dims),node_num,node_num)) 
    for _index, dim in enumerate(embedding_dims):
      #print (dim)
      model = Word2Vec(walks, size=dim,window=args.window_size, min_count=0, sg=1, workers=args.workers, iter=args.iter)    
      emb_matrix = np.zeros((node_num,dim))      
      for _cnt,node in enumerate(G.nodes()):
        emb_matrix[_cnt,:] = model[str(node)] 
      emb_matrix = emb_matrix - np.mean(emb_matrix,axis=0) 
      cosine_matrix = 1 - cdist(emb_matrix,emb_matrix,'cosine')
      if _index == 0:
        benchmark_array = np.array(upper_tri_masking(cosine_matrix))
        #np.savez_compressed('./pic/conect_data/npz/{}'.format(str.split(args.input,'/')[6]),benchmark_array)      
      else:
        dim_array = np.array(upper_tri_masking(cosine_matrix)) 
        loss = np.linalg.norm((dim_array-benchmark_array),ord=1)
        norm_loss.append(loss/len(dim_array))
    return embedding_dims[1:],norm_loss
    
def upper_tri_masking(A):
    '''
    Masking the upper triangular matrix. 
    '''
    m = A.shape[0]
    r = np.arange(m)
    mask = r[:,None] < r
    return A[mask]
  
def cal_embedding_distance(args):
    '''
    The overall random walk, graph embedding and cosine distance calculation process.
    '''
    nx_G = read_graph(args)
    G = node2vec.Graph(nx_G, args.directed, args.p, args.q)
    G.preprocess_transition_probs()
    walks = G.simulate_walks(args.num_walks, args.length)
    dims, loss = cal_cosine_matrices(nx_G,walks,args)
    plt.plot(dims,loss)
    plt.savefig('./a.png')
    identify_optimal_dim(dims, loss)
    #return cosine_matrices


