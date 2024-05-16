"""
Hard-Negative NCE loss for contrastive learning.
https://arxiv.org/pdf/2301.02280.pdf
"""
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class HardNegativeNCE(nn.Module):
    """
    Hard Negative NCE loss for contrastive learning.
    """
    def __init__(self, temperature: float = 0.07, alpha: float = 1.0, beta: float = 0.0):
        """
        Args:
            temperature: temperature for the softmax
            alpha: rescaling factor for positiver terms
            beta: concentration parameter

        Note:
            alpha = 1 and beta = 0 corresponds to the original Info-NCE loss
        """
        super(HardNegativeNCE, self).__init__()
        self.temperature = temperature
        self.alpha = alpha
        self.beta = beta

    def forward(
        self,
        video_embds: torch.Tensor,
        text_embds: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        debug_test: bool = False,
    ) -> float:
        """
        Args:
            video_embds: (batch_size, video_embd_dim)
            text_embds: (batch_size, text_embd_dim)
            debug_test: if True, then also compute Info-NCE loss
        """
        batch_size = video_embds.size(0)
        # computation of the similarity matrix
        sim_matrix = video_embds @ text_embds.T  # (batch_size, batch_size)
        # scale the similarity matrix with the temperature
        sim_matrix = sim_matrix / self.temperature
        sim_matrix = sim_matrix.float()
        if labels is not None:
            mask = (labels.unsqueeze(0) == labels.unsqueeze(1))
            mask = mask & (~torch.eye(
                len(sim_matrix),
                device=sim_matrix.device,
                dtype=mask.dtype
            )).bool()
            sim_matrix = sim_matrix.masked_fill(mask, float('-inf'))

        # sim_matrix(i, j) = <v_i, t_j>

        nominator = torch.diagonal(sim_matrix)
        if debug_test:
            # V2T
            denominator = torch.logsumexp(sim_matrix, dim=1)

            # T2V
            denominator2 = torch.logsumexp(sim_matrix, dim=0)
        beta_sim = self.beta * sim_matrix
        w_v2t = (batch_size - 1) * torch.exp(beta_sim) / \
            (torch.exp(beta_sim).sum(dim=1) - torch.exp(torch.diagonal(beta_sim)))
        w_t2v = (batch_size - 1) * torch.exp(beta_sim) / \
            (torch.exp(beta_sim).sum(dim=0) - torch.exp(torch.diagonal(beta_sim)))
        # replace the diagonal terms of w_v2t and w_t2v with alpha
        w_v2t[range(batch_size), range(batch_size)] = self.alpha
        w_t2v[range(batch_size), range(batch_size)] = self.alpha
        denominator_v2t = torch.log((torch.exp(sim_matrix) * w_v2t).sum(dim=1))
        denominator_t2v = torch.log((torch.exp(sim_matrix) * w_t2v).sum(dim=0))
        hn_nce_loss = (denominator_v2t - nominator).mean() + (denominator_t2v - nominator).mean()
        if debug_test:
            info_nce_loss = (denominator - nominator).mean() + (denominator2 - nominator).mean()
            print(f"hn_nce_loss: {hn_nce_loss}")
            print(f"info_nce_loss: {info_nce_loss}")
        return hn_nce_loss


if __name__ == "__main__":
    # sanity check that the loss is working
    # looking whether the loss is equal to the Info-NCE loss
    # when alpha = 1 and beta = 0
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for _ in range(10):
        in_video_embd = torch.randn(1024, 512).to(device)
        in_text_embd = torch.randn(1024, 512).to(device)

        # normalize
        in_video_embd = F.normalize(in_video_embd, dim=-1)
        in_text_embd = F.normalize(in_text_embd, dim=-1)
        loss_fn = HardNegativeNCE(beta=0.0, alpha=1.0)
        loss_fn(in_video_embd, in_text_embd, debug_test=True)
