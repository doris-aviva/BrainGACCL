import random

import h5py
import pandas as pd

import torch
import scipy.io
import numpy as np
from torch_geometric.data import Data

from option import parse

argv = parse()


def batch_adj(x):
    mean_x = torch.mean(x, 1, keepdim=True)
    xm = x.sub(mean_x.expand_as(x))
    c = xm.mm(xm.t())
    c = c / (x.size(1) - 1)
    d = torch.diag(c)
    stddev = torch.pow(d, 0.5)
    c = c.div(stddev.expand_as(c))
    c = c.div(stddev.expand_as(c).t())
    c = torch.clamp(c, -1.0, 1.0)
    return c


# sparsity
class Percentile(torch.autograd.Function):
    def __init__(self):
        super().__init__()

    def __call__(self, input, percentiles):
        return self.forward(input, percentiles)

    def forward(self, input, percentiles):
        input = torch.flatten(input) # find percentiles for flattened axis
        input_dtype = input.dtype
        input_shape = input.shape
        if isinstance(percentiles, int):
            percentiles = (percentiles,)
        if not isinstance(percentiles, torch.Tensor):
            percentiles = torch.tensor(percentiles, dtype=torch.double)
        if not isinstance(percentiles, torch.Tensor):
            percentiles = torch.tensor(percentiles)
        input = input.double()
        percentiles = percentiles.to(input.device).double()
        input = input.view(input.shape[0], -1)
        in_sorted, in_argsort = torch.sort(input, dim=0)
        positions = percentiles * (input.shape[0]-1) / 100
        floored = torch.floor(positions)
        ceiled = floored + 1
        ceiled[ceiled > input.shape[0] - 1] = input.shape[0] - 1
        weight_ceiled = positions-floored
        weight_floored = 1.0 - weight_ceiled
        d0 = in_sorted[floored.long(), :] * weight_floored[:, None]
        d1 = in_sorted[ceiled.long(), :] * weight_ceiled[:, None]
        self.save_for_backward(input_shape, in_argsort, floored.long(),
                               ceiled.long(), weight_floored, weight_ceiled)
        result = (d0+d1).view(-1, *input_shape[1:])
        return result.type(input_dtype)

    def backward(self, grad_output):
        (input_shape, in_argsort, floored, ceiled,
         weight_floored, weight_ceiled) = self.saved_tensors

        cols_offsets = (
            torch.arange(
                    0, input_shape[1], device=in_argsort.device)
            )[None, :].long()
        in_argsort = (in_argsort*input_shape[1] + cols_offsets).view(-1).long()
        floored = (
            floored[:, None]*input_shape[1] + cols_offsets).view(-1).long()
        ceiled = (
            ceiled[:, None]*input_shape[1] + cols_offsets).view(-1).long()

        grad_input = torch.zeros((in_argsort.size()), device=self.device)
        grad_input[in_argsort[floored]] += (grad_output
                                            * weight_floored[:, None]).view(-1)
        grad_input[in_argsort[ceiled]] += (grad_output
                                           * weight_ceiled[:, None]).view(-1)

        grad_input = grad_input.view(*input_shape)
        return grad_input


def construct_adjacent_matrix(pc, sparsity):
    num_elements = pc.numel()
    num_nonzero = int(num_elements * (sparsity / 100))

    flat_pc = pc.flatten()
    sorted_indices = torch.argsort(flat_pc, descending=True)

    adj_matrix = torch.zeros_like(pc)

    top_indices = sorted_indices[:num_nonzero]
    adj_matrix = adj_matrix.view(-1).scatter_(0, top_indices, 1).view_as(pc)

    _i = adj_matrix.nonzero(as_tuple=False).T
    _v = torch.ones(_i.size(1), device=pc.device)

    return torch.sparse.FloatTensor(_i, _v, pc.size())


def adjacent_matrix(x, sparsity, self_loop=False):
    if self_loop:
        a = construct_adjacent_matrix(x, sparsity)
    else:
        a = construct_adjacent_matrix(x - torch.eye(x.shape[0], x.shape[1]), sparsity).to_dense() + torch.eye(x.shape[0], x.shape[1])

    return a


def get_dataset():
    dataset = []
    data = Data()
    y = torch.tensor(0)
    data_nyu = scipy.io.loadmat(argv.data_path)
    bold = data_nyu['AAL'][0]
    labels = data_nyu['lab'][0]

    for i in range(argv.num_graphs):
        bold_i = bold[i]
        x = torch.tensor(bold_i[:][:]).float()
        x = torch.transpose(x, 0, 1)

        if torch.isnan(x).any() or torch.isinf(x).any():
            raise ValueError(f"The input data contains NaN or infinity at the index {i}")

        edge_index = torch.zeros((2, argv.num_nodes * argv.num_nodes))
        edge_weight_array = torch.zeros((1, argv.num_nodes * argv.num_nodes))
        A = np.corrcoef(x)
        A = np.nan_to_num(A, nan=0.0, posinf=0.0, neginf=0.0)
        A_torch = torch.tensor(A, dtype=torch.float)
  
        AS = adjacent_matrix(A_torch, sparsity=argv.sparsity, self_loop=argv.self_loop)

        x = A_torch

        edge_index = AS._indices()
        edge_weight_array = AS._values()
        edge_weight_array = edge_weight_array.unsqueeze(0)

        y = torch.tensor(labels[i])
        y = y.to(torch.long)
        data = Data(x=x, edge_index=edge_index, edge_weight=edge_weight_array)
        data.y = y
        dataset.append(data)

    print("data import complete\n")

    return dataset, labels

