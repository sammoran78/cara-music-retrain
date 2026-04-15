from __future__ import annotations

from typing import Iterable

import torch


class ConstrainedCARADecoder:
    def __init__(self, codebook) -> None:
        self.codebook = codebook

    def decode(self, cw_logits: Iterable[torch.Tensor], prob_bins: torch.Tensor) -> list[str]:
        batch_size = prob_bins.shape[0]
        results: list[str] = []
        cw_logits = list(cw_logits)
        for batch_index in range(batch_size):
            slots: list[str] = []
            seen_codewords: set[str] = set()
            for slot_index in range(len(cw_logits)):
                logits = cw_logits[slot_index][batch_index].clone()
                while True:
                    cw_idx = int(logits.argmax().item())
                    codeword = self.codebook.idx_to_codeword[cw_idx]
                    if codeword not in seen_codewords:
                        break
                    logits[cw_idx] = float("-inf")
                seen_codewords.add(codeword)
                probability = int(prob_bins[batch_index, slot_index].item())
                slots.append(f"{codeword}@{probability:02d}")
            results.append("ATTR|" + "|".join(slots) + "|END")
        return results

    def validate_format(self, attr_string: str) -> tuple[bool, str]:
        if not attr_string.startswith("ATTR|") or not attr_string.endswith("|END"):
            return False, "Missing ATTR/END delimiters"
        body = attr_string[5:-4]
        slots = body.split("|")
        if len(slots) != 3:
            return False, f"Expected 3 slots, got {len(slots)}"
        total_prob = 0
        for slot in slots:
            if "@" not in slot:
                return False, f"Missing @ separator in slot: {slot}"
            codeword, probability = slot.rsplit("@", 1)
            if not self.codebook.is_registered(codeword):
                return False, f"Unregistered codeword: {codeword}"
            if not self.codebook.checksum_valid(codeword):
                return False, f"Invalid checksum: {codeword}"
            if not probability.isdigit() or len(probability) != 2:
                return False, f"Invalid probability bin: {probability}"
            total_prob += int(probability)
        if total_prob != 100:
            return False, f"Probabilities sum to {total_prob}, expected 100"
        return True, "Valid"
