import copy

import torch
from torch.nn import Sequential, Linear, ReLU


class StructureAugmentor(torch.nn.Module):
	def __init__(self, encoder, mlp_dim=64):
		super(StructureAugmentor, self).__init__()

		self.encoder = encoder
		self.input_dim = self.encoder.out_dim

		self.mlp_model = Sequential(
			Linear(self.input_dim * 2, mlp_dim),
			ReLU(),
			Linear(mlp_dim, 1)
		)
		self.init_emb()

	def init_emb(self):
		for m in self.modules():
			if isinstance(m, Linear):
				torch.nn.init.xavier_uniform_(m.weight.data)
				if m.bias is not None:
					m.bias.data.fill_(0.0)

	def forward(self, data_in, device):
		data = copy.deepcopy(data_in)
		batch = data.batch
		num_graphs = data.num_graphs
		x = data.x
		edge_index = data.edge_index
		edge_weight = data.edge_weight
		_, node_embed = self.encoder(batch, x, edge_index, None, edge_weight)

		source_nodes, target_nodes = edge_index[0], edge_index[1]
		source_embed = node_embed[source_nodes]
		target_embed = node_embed[target_nodes]

		edge_embed = torch.cat([source_embed, target_embed], dim=1)
		edge_logits = self.mlp_model(edge_embed)

		temperature = 1.0
		bias = 0.0 + 0.0001
		eps = (bias - (1 - bias)) * torch.rand(edge_logits.size()) + (1 - bias)
		noise = torch.log(eps) - torch.log(1 - eps)
		noise = noise.to(device)
		logits = (noise + edge_logits) / temperature
		aug_edge_weight = torch.sigmoid(logits).squeeze()
		aug_edge_weight = aug_edge_weight.view(num_graphs, -1)
		data.edge_weight = torch.mul(edge_weight, aug_edge_weight)
		return aug_edge_weight, data