
import torch
import torch.nn as nn
from torch.nn import Sequential as Seq, Linear, ReLU
from torch_scatter import scatter_mean
from torch_geometric.nn import MetaLayer

from abc import ABC, abstractmethod
import itertools

# My libraries
import graph_construction as gc

dont_send_to_device = []

class EdgeModel(torch.nn.Module):
    def __init__(self, config):
        super(EdgeModel, self).__init__()

        in_channels = config['e_inc'] + 2*config['n_inc'] + config['u_inc']
        hs1 = config['edge_model_mlp1_hidden_sizes'][0]
        hs2 = config['edge_model_mlp1_hidden_sizes'][1]

        self.edge_mlp = Seq(Linear(in_channels, hs1), 
                            ReLU(), 
                            Linear(hs1, hs2),
                            ReLU(),
                            Linear(hs2, config['e_outc']))

    def forward(self, src, dest, edge_attr, u, batch):
        """ Edge Model of Graph Net layer

            @param src: [E x n_inc], where E is the number of edges. Treat E as batch size
            @param dest: [E x n_inc], where E is the number of edges.
            @param edge_attr: [E x e_inc]
            @param u: [B x u_inc], where B is the number of graphs.
            @param batch: [E] with max entry B - 1.

            @return a [E x e_outc] torch tensor
        """

        src_dest_edge_u = torch.cat([src, dest, edge_attr, u[batch]], 1) # Shape: [E x (2*n_inc + e_inc + u_inc)]
        out = self.edge_mlp(src_dest_edge_u) # Shape: [E x e_outc]
        return out

class NodeModel(torch.nn.Module):
    def __init__(self, config):
        super(NodeModel, self).__init__()

        mlp1_inc = config['n_inc'] + config['e_outc']
        mlp1_hs1 = config['node_model_mlp1_hidden_sizes'][0]
        mlp1_hs2 = config['node_model_mlp1_hidden_sizes'][1]

        mlp2_inc = config['n_inc'] + mlp1_hs2 + config['u_inc']
        mlp2_hs1 = config['node_model_mlp2_hidden_sizes'][0]

        self.node_mlp_1 = Seq(Linear(mlp1_inc, mlp1_hs1), 
                              ReLU(), 
                              Linear(mlp1_hs1, mlp1_hs2))
        self.node_mlp_2 = Seq(Linear(mlp2_inc, mlp2_hs1), 
                              ReLU(), 
                              Linear(mlp2_hs1, config['n_outc']))

    def forward(self, x, edge_index, edge_attr, u, batch):
        """ Node Model of Graph Net layer

            @param x: [N x n_inc], where N is the number of nodes.
            @param edge_index: [2 x E] with max entry N - 1.
            @param edge_attr: [E x e_inc]
            @param u: [B x u_inc]
            @param batch: [N] with max entry B - 1.

            @return: a [N x n_outc] torch tensor
        """

        row, col = edge_index

        srcnode_edge = torch.cat([x[row], edge_attr], dim=1) # Concat source node features, edge features. Shape: [E x (n_inc + e_outc)]
        srcnode_edge = self.node_mlp_1(srcnode_edge) # Run this through an MLP... Shape: [E x hs2]
        per_node_aggs = scatter_mean(srcnode_edge, col, dim=0, dim_size=x.size(0)) # Mean-aggregation for every node (dest node for an edge). Shape: [N x hs2]

        node_agg_u = torch.cat([x, per_node_aggs, u[batch]], dim=1) # Concat node, aggregated edge/src-node features, u
        out = self.node_mlp_2(node_agg_u) # Run through MLP. Shape: [N x n_outc]

        return out

class GlobalModel(torch.nn.Module):
    def __init__(self, config):
        super(GlobalModel, self).__init__()

        inc = config['u_inc'] + config['n_outc'] + config['e_outc']
        hs1 = config['global_model_mlp1_hidden_sizes'][0]

        self.global_mlp = Seq(Linear(inc, hs1), 
                              ReLU(), 
                              Linear(hs1, config['u_outc']))

    def forward(self, x, edge_index, edge_attr, u, batch):
        """ Global Update of Graph Net Layer

            @param x: [N x n_outc], where N is the number of nodes.
            @param edge_index: [2 x E] with max entry N - 1.
            @param edge_attr: [E x e_outc]
            @param u: [B x u_inc]
            @param batch: [N] with max entry B - 1.

            @return: a [B x u_outc] torch tensor
        """

        row, col = edge_index
        edge_batch = batch[row] # edge_batch is same as batch in EdgeModel.forward(). Shape: [E]

        per_batch_edge_aggregations = scatter_mean(edge_attr, edge_batch, dim=0) # Shape: [B x e_outc]
        per_batch_node_aggregations = scatter_mean(x, batch, dim=0) # Shape: [B x n_outc]

        out = torch.cat([u, per_batch_node_aggregations, per_batch_edge_aggregations], dim=1) # Shape: [B x (u_inc + n_outc + e_outc)]
        return self.global_mlp(out)

class GraphNet(torch.nn.Module):

    def __init__(self, layer_config):
        """

            @param layer_config: a list of dictionaries specifying GraphNet layers
        """
        super(GraphNet, self).__init__()

        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.layer_config = layer_config

        # Build the graph network layers
        self.build_network()

    def build_network(self):

        self.layers = []
        for lp in self.layer_config:
            gn_layer = MetaLayer(EdgeModel(lp), NodeModel(lp), GlobalModel(lp))
            gn_layer.to(self.device)
            self.layers.append(gn_layer)
        self.layers = torch.nn.ModuleList(self.layers)

    def forward(self, g):
        """ Forward pass of GraphNet

            @param g: a graph (torch_geometric.Data) instance. (batched)

            @return: a graph (torch_geometric.Data) instance. (batched)
        """

        x, edge_attr, u = g.x, g.edge_attr, g.u
        for layer in self.layers:
            x, edge_attr, u = layer(x, g.edge_index, edge_attr, u, g.batch)

        # Clone the torch_geometric.Data object. This means x/edge_attr/u no longer have gradients
        #   But, replace with computed x/edge_attr/u which does have gradients
        new_g = g.clone()
        new_g.x = x + g.x # residual
        new_g.edge_attr = edge_attr + g.edge_attr
        new_g.u = u + g.u

        return new_g



class NetworkWrapper(ABC):

    def __init__(self, config):
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.config = config.copy()

        # Build network and losses
        self.setup()

    @abstractmethod
    def setup(self):
        pass

    def train_mode(self):
        """ Put all modules into train mode
        """
        self.model.train()

    def eval_mode(self):
        """ Put all modules into eval mode
        """
        self.model.eval()

    def send_batch_to_device(self, batch):
        for key in batch.keys():
            if key in dont_send_to_device:
                continue
            if len(batch[key]) == 0: # can happen if a modality (e.g. RGB) is not loaded
                continue
            batch[key] = batch[key].to(self.device)

    def save(self, filename):
        """ Save the model as a checkpoint
        """
        checkpoint = {'model' : self.model.state_dict()}
        torch.save(checkpoint, filename)

    def load(self, filename):
        """ Load the model checkpoint
        """
        checkpoint = torch.load(filename)
        self.model.load_state_dict(checkpoint['model'])
        print(f"Loaded {self.__class__.__name__} model")


class GraphNetWrapper(NetworkWrapper):

    def setup(self):
        """ Setup model, losses, optimizers, misc
        """

        # Whole model, for nn.DataParallel
        self.model = GraphNet(self.config['layer_config'])
        self.model.to(self.device)
        
    def run_on_batch(self, batch):
        """ Run algorithm on batch of images in eval mode
        
            @param batch: A Python dictionary with keys:
                            'current_state' : a [B x n x d] torch tensor
                            'next_state' : a [B x n x d] torch tensor
        """

        self.eval_mode()
        self.send_batch_to_device(batch)

        curr_graph = gc.batch_construct_billiards_graph(batch['current_state'])
        with torch.no_grad():
            pred_next_graph = self.model(curr_graph)

        return pred_next_graph

    def rollout(self, s0, T):
        """ Run sequence starting from state 0 (s0) for T steps

            @param x: s0 [n x d] torch tensor. 
            @param T: Number of time steps to rollout
        """
        n, d = s0.shape

        states = torch.zeros((T, n, d), dtype=torch.float, device=s0.device)
        states[0] = s0.clone()

        for t in range(1, T):
            pred_next_graph = self.run_on_batch({'current_state' : states[t-1:t]})
            states[t] = pred_next_graph.x.clone()

        return states
