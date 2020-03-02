import sys, os
from itertools import permutations
import torch
import numpy as np

from torch_geometric.data import Data, Batch


def batch_construct_billiards_graph(x):
    """ Construct Graph from Billiards data

        Node features are simply states
        Edge connections are fully connected
        Edge features are center offsets
        Global attribute u starts at 0

        @param x: a [B x n x d] torch tensor. 
                  Batch size B, n objects, each row is [x,y,dx,dy]

        @return: a Data instance with Data.x.shape = [Bn x d]
    """
    B, n, d = x.shape

    # Compute node features for all graphs
    # DONE, x is already the node features

    # Compute edge connections
    edge_index = torch.tensor(list(permutations(range(n), 2)), dtype=torch.long)
    edge_index = edge_index.t().contiguous() # Shape: [2 x E], E = n^2

    # Compute edge features for all graphs
    src, dest = edge_index
    edge_attrs = x[:, dest, :2] - x[:, src, :2] # Shape: [B x E x 2]

    # U vector. 4-dim 0-vector
    u = torch.zeros((1,4), dtype=torch.float, device=x.device)

    # Create list of Data objects, then call Batch.from_data_list()
    data_objs = [Data(x=x[b], 
                      edge_index=edge_index, 
                      edge_attr=edge_attrs[b],
                      u=u.clone()
                     ) 
                 for b in range(B)]
    batch = Batch.from_data_list(data_objs).to(x.device)

    return batch