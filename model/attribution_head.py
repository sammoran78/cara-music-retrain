from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class CARAAttributionHead(nn.Module):
    def __init__(self, dit_hidden_dim: int, num_codewords: int, num_slots: int = 3) -> None:
        super().__init__()
        self.num_slots = num_slots
        self.num_codewords = num_codewords
        self.feature_net = nn.Sequential(
            nn.Linear(dit_hidden_dim, 512),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Dropout(0.1),
        )
        self.cw_heads = nn.ModuleList([nn.Linear(256, num_codewords) for _ in range(num_slots)])
        self.prob_head = nn.Linear(256, num_slots)

    def forward(self, dit_hidden_states: torch.Tensor):
        features = dit_hidden_states.mean(dim=1)
        features = self.feature_net(features)
        cw_logits = [head(features) for head in self.cw_heads]
        prob_logits = self.prob_head(features)
        prob_dist = F.softmax(prob_logits, dim=-1)
        prob_bins = self._to_bins(prob_dist)
        return cw_logits, prob_dist, prob_bins

    def _to_bins(self, prob_dist: torch.Tensor) -> torch.Tensor:
        bins = (prob_dist * 100).round().long()
        remainder = 100 - bins.sum(dim=-1, keepdim=True)
        bins[:, 0] += remainder.squeeze(-1)
        return bins
