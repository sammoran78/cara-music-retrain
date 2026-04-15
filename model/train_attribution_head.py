from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, random_split

from model.attribution_head import CARAAttributionHead


@dataclass
class SoftTargetExample:
    source_id: str
    codeword_indices: list[int]
    probabilities: list[float]


class AttributionDataset(Dataset):
    def __init__(self, dit_hidden_states_dir: str | Path, soft_targets_path: str | Path, codebook) -> None:
        self.hidden_dir = Path(dit_hidden_states_dir)
        self.codebook = codebook
        self.examples = self._load_examples(Path(soft_targets_path))
        sample_hidden = self._load_hidden(self.examples[0].source_id)
        self.hidden_dim = int(sample_hidden.shape[-1])

    def _load_examples(self, path: Path) -> list[SoftTargetExample]:
        examples: list[SoftTargetExample] = []
        with path.open("r", encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                raw_targets = json.loads(row["soft_targets"])
                indices = []
                probabilities = []
                for target in raw_targets[:3]:
                    codeword = target.get("codeword") or target.get("pool_id")
                    if codeword in self.codebook.codeword_to_idx:
                        indices.append(self.codebook.codeword_to_idx[codeword])
                    else:
                        indices.append(0)
                    probabilities.append(float(target.get("probability", 0)))
                while len(indices) < 3:
                    indices.append(0)
                    probabilities.append(0.0)
                examples.append(SoftTargetExample(row["source_id"], indices, probabilities))
        return examples

    def _load_hidden(self, source_id: str) -> torch.Tensor:
        path = self.hidden_dir / f"{source_id}.npy"
        if not path.exists():
            layer_matches = sorted(self.hidden_dir.glob(f"{source_id}_layer*.npy"))
            if not layer_matches:
                raise FileNotFoundError(f"No hidden states found for {source_id}")
            path = layer_matches[-1]
        return torch.tensor(np.load(path), dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int):
        example = self.examples[index]
        hidden = self._load_hidden(example.source_id)
        return hidden, torch.tensor(example.codeword_indices), torch.tensor(example.probabilities, dtype=torch.float32)


def cara_loss(cw_logits, prob_dist, target_cw_indices, target_prob_dist, num_slots: int = 3):
    cw_loss = sum(F.cross_entropy(cw_logits[slot], target_cw_indices[:, slot]) for slot in range(num_slots)) / num_slots
    pred_log = torch.log(prob_dist + 1e-8)
    target_normed = target_prob_dist / (target_prob_dist.sum(dim=-1, keepdim=True) + 1e-8)
    prob_loss = F.kl_div(pred_log, target_normed, reduction="batchmean")
    return cw_loss + 0.5 * prob_loss


def evaluate_head(head, val_loader, device: str = "cpu") -> dict[str, float]:
    losses = []
    with torch.no_grad():
        for hidden_states, target_cw_indices, target_probs in val_loader:
            hidden_states = hidden_states.to(device)
            target_cw_indices = target_cw_indices.to(device)
            target_probs = target_probs.to(device)
            cw_logits, prob_dist, _prob_bins = head(hidden_states)
            losses.append(float(cara_loss(cw_logits, prob_dist, target_cw_indices, target_probs).item()))
    return {"val_loss": float(np.mean(losses)) if losses else 0.0}


def train_attribution_head(
    dit_hidden_states_dir,
    soft_targets_path,
    codebook,
    num_epochs: int = 50,
    batch_size: int = 256,
    learning_rate: float = 1e-3,
    device: str = "cpu",
    checkpoint_path: str | Path = "checkpoints/attribution_head_v1.pt",
):
    dataset = AttributionDataset(dit_hidden_states_dir, soft_targets_path, codebook)
    train_size = max(1, int(len(dataset) * 0.9))
    val_size = max(1, len(dataset) - train_size)
    if train_size + val_size > len(dataset):
        train_size = len(dataset) - val_size
    train_set, val_set = random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size)
    head = CARAAttributionHead(dataset.hidden_dim, len(codebook.codeword_to_idx), num_slots=3).to(device)
    optimizer = torch.optim.Adam(head.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, num_epochs))
    history: list[dict[str, Any]] = []
    for epoch in range(num_epochs):
        head.train()
        losses = []
        for hidden_states, target_cw_indices, target_probs in train_loader:
            hidden_states = hidden_states.to(device)
            target_cw_indices = target_cw_indices.to(device)
            target_probs = target_probs.to(device)
            cw_logits, prob_dist, _prob_bins = head(hidden_states)
            loss = cara_loss(cw_logits, prob_dist, target_cw_indices, target_probs)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(float(loss.item()))
        val_metrics = evaluate_head(head, val_loader, device=device)
        scheduler.step()
        history.append({"epoch": epoch, "train_loss": float(np.mean(losses)) if losses else 0.0, **val_metrics})
    checkpoint = Path(checkpoint_path)
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save(head.state_dict(), checkpoint)
    return head, history
