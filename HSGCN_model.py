import torch
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree

class HSGCN(MessagePassing):
    def __init__(self):
        super(HSGCN, self).__init__(aggr='add')

    def forward(self, x, edge_index):
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
        m = self.propagate(edge_index, x=x)
        d = torch.mul(m, x)
        c = -F.relu(2 * -d) + 1
        return torch.mul(c, x)

    def update(self, aggr_out, x):
        return torch.clamp(aggr_out.float() + 2 * x, -1, 1)
