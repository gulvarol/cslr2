"""
Used in DDP mode with the HN-NCE loss.
"""
import torch
import torch.distributed as dist


class DiffAllGather(torch.autograd.Function):
    """Gathering all tensors from all processes"""
    @staticmethod
    def forward(ctx, tensor):
        """
        Forward pass with gathering all tensors
        """
        gathered = [torch.zeros_like(tensor) for _ in range(dist.get_world_size())]
        dist.all_gather(gathered, tensor)
        return tuple(gathered)

    @staticmethod
    def backward(ctx, *grad_outs):
        """
        Backward pass with all-reduce
		"""
        grad_outs = torch.stack(grad_outs)
        dist.all_reduce(grad_outs)
        return grad_outs[dist.get_rank()]


def all_gather(tensor):
    """All gather tensors from all processes"""
    return DiffAllGather.apply(tensor)
