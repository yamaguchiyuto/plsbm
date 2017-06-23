import sys
import argparse

import numpy as np 
from scipy.sparse import lil_matrix

from lib.plsbm_vem import PLSBM_VEM
from lib.sbm_vem import SBM_VEM

def load_labels(filepath):
    assign = np.genfromtxt(filepath, dtype=int)
    nnodes = assign.shape[0]
    return (assign, nnodes)

def load_graph(filepath, nnodes):
    G = lil_matrix((nnodes, nnodes))
    for line in open(filepath):
        src,dst = map(int, line.strip().split(' '))
        G[src,dst] = 1
        G[dst,src] = 1
    return G.tocsr()

def dump_result(outfile, result):
    np.savetxt(outfile, result, fmt='%1.0f')

p = argparse.ArgumentParser()
p.add_argument("-g", "--graphfile", help="input graph file", type=str, required=True)
p.add_argument("-l", "--labelfile", help="input label file", type=str, required=True)
p.add_argument("-o", "--outfile", help="output file", type=str, required=True)
p.add_argument("-m", "--method", help="plsbm or sbm", type=str, required=True)
p.add_argument("-v", "--verbose", help="verbosity", action='store_true')
p.add_argument("-e", "--eps", help="threshold of residual error for convergence", type=float, default=0.001)
args = p.parse_args()

graphfile = args.graphfile
labelfile = args.labelfile
outfile = args.outfile
alg = args.method
verbose = args.verbose
eps = args.eps

y, nnodes = load_labels(labelfile)
A = load_graph(graphfile, nnodes)

if alg == 'plsbm':
    model = PLSBM_VEM(A, y, verbose, eps)
elif alg == 'sbm':
    k = y.max() + 1
    model = SBM_VEM(A, k, verbose, eps)

model.fit()
result = model.predict()
dump_result(outfile, result)
