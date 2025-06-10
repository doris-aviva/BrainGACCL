import copy

import torch
import torch.nn.functional as F
from torch.nn import Sequential, Linear, ReLU

from torch_geometric.nn import VGAE
from torch import Tensor

from typing import List, Optional, Tuple, Union
from torch_geometric.typing import OptTensor
from torch_geometric.utils.map import map_index
from torch_geometric.utils.mask import index_to_mask
from torch_geometric.utils.num_nodes import maybe_num_nodes

from GINlayer import WGIN


def subgraph(
    subset: Union[Tensor, List[int]],
    edge_index: Tensor,
    num_graphs: int,
    edge_weight: OptTensor = None,
    relabel_nodes: bool = False,
    num_nodes: Optional[int] = None,
    return_edge_mask: bool = False,
) -> Union[Tuple[Tensor, OptTensor], Tuple[Tensor, OptTensor, OptTensor]]:

    device = edge_index.device

    if isinstance(subset, (list, tuple)):
        subset = torch.tensor(subset, dtype=torch.long, device=device)

    if subset.dtype != torch.bool:
        num_nodes = maybe_num_nodes(edge_index, num_nodes)
        node_mask = index_to_mask(subset, size=num_nodes)
    else:
        num_nodes = subset.size(0)
        node_mask = subset
        subset = node_mask.nonzero().view(-1)

    edge_mask = node_mask[edge_index[0]] & node_mask[edge_index[1]]
    edge_index = edge_index[:, edge_mask]
    edge_mask = edge_mask.unsqueeze(0)
    edge_mask = edge_mask.view(num_graphs, -1)
    edge_mask_float = edge_mask.float()
    edge_weight_float = torch.mul(edge_weight, edge_mask_float)
    edge_weight = edge_weight[edge_mask] if edge_weight is not None else None

    if relabel_nodes:
        edge_index, _ = map_index(
            edge_index.view(-1),
            subset,
            max_index=num_nodes,
            inclusive=True,
        )
        edge_index = edge_index.view(2, -1)

    if return_edge_mask:
        return edge_index, edge_weight, edge_mask
    else:
        return edge_index, edge_weight, edge_weight_float


class AttributeAugmentor(VGAE):
    def __init__(self, num_features, dim, encoder, add_mask=True):
        self.add_mask = add_mask
        encoder = encoder(num_features, dim, self.add_mask)
        super().__init__(encoder=encoder)

    def forward(self, data_in, requires_grad):
        data = copy.deepcopy(data_in)

        x, edge_index = data.x, data.edge_index
        num_graphs = data.num_graphs
        edge_weight = None
        if data.edge_weight is not None:
            edge_weight = data.edge_weight

        data.x = data.x.float()
        x = x.float()
        x.requires_grad = requires_grad

        p = self.encoder(data)
        sample = F.gumbel_softmax(p, hard=True)
        
        real_sample = sample[:, 0]
        attr_mask_sample = None
        if self.add_mask == True:
            attr_mask_sample = sample[:, 2]
            keep_sample = real_sample + attr_mask_sample
        else:
            keep_sample = real_sample

        keep_idx = torch.nonzero(keep_sample, as_tuple=False).view(-1, )
        edge_index, edge_weight, edge_weight_float = subgraph(keep_idx, edge_index, num_graphs, edge_weight, num_nodes=data.num_nodes)
        x = x * keep_sample.view(-1, 1)

        if self.add_mask == True:
            attr_mask_idx = attr_mask_sample.bool()
            token = data.x.detach().mean()
            x[attr_mask_idx] = token

        data.x = x
        data.edge_index = edge_index
        if data.edge_weight is not None:
            data.edge_weight = edge_weight

        return keep_sample, data, edge_weight_float
