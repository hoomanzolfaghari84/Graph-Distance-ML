"""
Modernized implementation of Sparsemax (Martins & Astudillo, 2016) for batched graph operations.
Original reference: https://github.com/OpenNMT/OpenNMT-py/blob/master/onmt/modules/sparse_activations.py
Paper: 'From Softmax to Sparsemax: A Sparse Model of Attention and Multi-Label Classification', ICML 2016

Usage:
>> x = torch.tensor([1.7301, 0.6792, -1.0565, 1.6614, -0.3196, -0.7790, -0.3877, -0.4943, 0.1831, -0.0061])
>> batch = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1, 1, 1])
>> sparse_attention = Sparsemax()
>> res = sparse_attention(x, batch)
>> print(res)
tensor([0.5343, 0.0000, 0.0000, 0.4657, 0.0612, 0.0000, 0.0000, 0.0000, 0.5640, 0.3748])
"""

import torch
import torch.nn as nn
from torch.autograd import Function
from torch_scatter import scatter

def scatter_sort(x, batch, fill_value=-1e16):
    """Sorts values within each batch group in descending order."""
    num_nodes = scatter(batch.new_ones(x.size(0)), batch, dim=0, reduce="sum")
    batch_size, max_num_nodes = num_nodes.size(0), num_nodes.max().item()
    cum_num_nodes = torch.cat([num_nodes.new_zeros(1), num_nodes.cumsum(dim=0)[:-1]], dim=0)

    # Compute flat indices for dense representation
    index = torch.arange(batch.size(0), dtype=torch.long, device=x.device)
    index = index - cum_num_nodes[batch] + (batch * max_num_nodes)

    # Create dense tensor and sort
    dense_x = torch.full((batch_size * max_num_nodes,), fill_value, dtype=x.dtype, device=x.device)
    dense_x[index] = x
    dense_x = dense_x.view(batch_size, max_num_nodes)
    sorted_x, _ = torch.sort(dense_x, dim=-1, descending=True)

    # Flatten and filter out fill values
    sorted_x = sorted_x.view(-1)
    mask = sorted_x != fill_value
    sorted_x = sorted_x[mask]
    cumsum_sorted_x = sorted_x.cumsum(dim=0)[mask]

    return sorted_x, cumsum_sorted_x


def _make_ix_like(batch):
    """Generates per-group indices (1, 2, ..., num_nodes) for each batch."""
    num_nodes = scatter(batch.new_ones(batch.size(0)), batch, dim=0, reduce="sum")
    idx = torch.arange(1, num_nodes.sum() + 1, dtype=torch.long, device=batch.device)
    splits = num_nodes.cumsum(dim=0)
    return torch.split(idx, splits[:-1].tolist())[0]


def _threshold_and_support(x, batch):
    """Computes Sparsemax threshold and support size per batch."""
    num_nodes = scatter(batch.new_ones(x.size(0)), batch, dim=0, reduce="sum")
    cum_num_nodes = torch.cat([num_nodes.new_zeros(1), num_nodes.cumsum(dim=0)[:-1]], dim=0)

    sorted_input, input_cumsum = scatter_sort(x, batch)
    input_cumsum = input_cumsum - 1.0
    rhos = _make_ix_like(batch).to(x.dtype)
    support = rhos * sorted_input > input_cumsum

    support_size = scatter(support.to(batch.dtype), batch, dim=0, reduce="sum")
    idx = support_size + cum_num_nodes - 1
    idx = idx.clamp(min=0)  # Handle invalid indices
    tau = input_cumsum.gather(0, idx)
    tau /= support_size.to(x.dtype)

    return tau, support_size


class SparsemaxFunction(Function):
    @staticmethod
    def forward(ctx, x, batch):
        """Sparsemax forward pass: applies sparse normalization per batch."""
        max_val = scatter(x, batch, dim=0, reduce="max")[batch]
        x_shifted = x - max_val  # Shift for numerical stability
        tau, supp_size = _threshold_and_support(x_shifted, batch)
        output = torch.clamp(x_shifted - tau[batch], min=0)
        ctx.save_for_backward(supp_size, output, batch)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        """Sparsemax backward pass."""
        supp_size, output, batch = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad_input[output == 0] = 0  # Zero gradient where output is zero

        v_hat = scatter(grad_input, batch, dim=0, reduce="sum") / supp_size.to(output.dtype)
        grad_input = torch.where(output != 0, grad_input - v_hat[batch], grad_input)

        return grad_input, None


sparsemax = SparsemaxFunction.apply


class Sparsemax(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, batch):
        return sparsemax(x, batch)


if __name__ == "__main__":
    sparse_attention = Sparsemax()
    input_x = torch.tensor([1.7301, 0.6792, -1.0565, 1.6614, -0.3196, -0.7790, -0.3877, -0.4943, 0.1831, -0.0061])
    input_batch = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1, 1, 1])
    res = sparse_attention(input_x, input_batch)
    print(res)