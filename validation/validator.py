from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import List


class AttrState(Enum):
    EXACT = "exact_valid"
    REPAIRED = "repaired"
    DEGRADED = "degraded_fallback"
    EXCEPTION = "exception"


@dataclass
class SlotResult:
    original_cw: str
    validated_cw: str
    probability: int
    state: AttrState
    repair_detail: str = ""


@dataclass
class ValidationResult:
    state: AttrState
    original_string: str
    validated_string: str
    slots: List[SlotResult] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    repairs: List[str] = field(default_factory=list)


class CARAValidator:
    def __init__(self, codebook, max_edit_distance: int = 1):
        self.codebook = codebook
        self.max_edit_distance = max_edit_distance

    def validate(self, attr_string: str) -> ValidationResult:
        structural_ok, structural_msg = self._check_structure(attr_string)
        if not structural_ok:
            return self._handle_structural_failure(attr_string, structural_msg)
        slots = self._parse_slots(attr_string)
        slot_results = [self._validate_slot(codeword, probability) for codeword, probability in slots]
        if sum(slot.probability for slot in slot_results) != 100:
            slot_results = self._normalise_probabilities(slot_results)
        overall_state = self._determine_state(slot_results)
        validated_string = self._assemble(slot_results)
        errors = [slot.repair_detail for slot in slot_results if slot.state != AttrState.EXACT]
        repairs = [slot.repair_detail for slot in slot_results if slot.state == AttrState.REPAIRED]
        return ValidationResult(
            state=overall_state,
            original_string=attr_string,
            validated_string=validated_string,
            slots=slot_results,
            errors=errors,
            repairs=repairs,
        )

    def _validate_slot(self, codeword: str, probability: int) -> SlotResult:
        if self.codebook.is_registered(codeword) and self.codebook.checksum_valid(codeword):
            return SlotResult(codeword, codeword, probability, AttrState.EXACT)
        if getattr(self.codebook, "is_registered_payload", None) and self.codebook.is_registered_payload(codeword):
            corrected = self.codebook.recompute_checksum(codeword)
            return SlotResult(codeword, corrected, probability, AttrState.REPAIRED, "checksum_recomputed")
        nearest, distance = self.codebook.nearest_codeword(codeword)
        if nearest is not None and distance <= self.max_edit_distance:
            alternatives = self.codebook.all_within_distance(codeword, distance)
            if len(alternatives) == 1:
                return SlotResult(codeword, nearest, probability, AttrState.REPAIRED, f"edit_distance_{distance}_unique")
        parent_codeword = self.codebook.get_parent_codeword(codeword)
        if parent_codeword:
            return SlotResult(codeword, parent_codeword, probability, AttrState.DEGRADED, f"parent_pool_fallback:{parent_codeword}")
        modality = codeword[0] if codeword else "M"
        root_codeword = self.codebook.get_root_codeword(modality)
        if root_codeword:
            return SlotResult(codeword, root_codeword, probability, AttrState.DEGRADED, f"root_pool_fallback:{root_codeword}")
        return SlotResult(codeword, "UNRESOLVED", probability, AttrState.EXCEPTION, f"all_validation_failed:original={codeword}")

    def _determine_state(self, slot_results: List[SlotResult]) -> AttrState:
        states = [slot.state for slot in slot_results]
        for worst in [AttrState.EXCEPTION, AttrState.DEGRADED, AttrState.REPAIRED]:
            if worst in states:
                return worst
        return AttrState.EXACT

    def _check_structure(self, attr_string: str) -> tuple[bool, str]:
        if not attr_string.startswith("ATTR|"):
            return False, "missing_ATTR_prefix"
        if not attr_string.endswith("|END"):
            return False, "missing_END_suffix"
        body = attr_string[5:-4]
        slots = body.split("|")
        if len(slots) != 3:
            return False, f"expected_3_slots_got_{len(slots)}"
        for slot in slots:
            if "@" not in slot:
                return False, f"missing_@_separator:{slot}"
        return True, "ok"

    def _parse_slots(self, attr_string: str) -> list[tuple[str, int]]:
        body = attr_string[5:-4]
        slots: list[tuple[str, int]] = []
        for slot in body.split("|"):
            codeword, probability = slot.rsplit("@", 1)
            slots.append((codeword, int(probability)))
        return slots

    def _normalise_probabilities(self, slot_results: List[SlotResult]) -> List[SlotResult]:
        total = sum(slot.probability for slot in slot_results)
        if total == 0:
            equal = 100 // max(1, len(slot_results))
            for slot in slot_results:
                slot.probability = equal
            slot_results[0].probability += 100 - sum(slot.probability for slot in slot_results)
            return slot_results
        for slot in slot_results:
            slot.probability = round(slot.probability / total * 100)
        slot_results[0].probability += 100 - sum(slot.probability for slot in slot_results)
        return slot_results

    def _assemble(self, slot_results: List[SlotResult]) -> str:
        slots_text = "|".join(f"{slot.validated_cw}@{slot.probability:02d}" for slot in slot_results)
        return f"ATTR|{slots_text}|END"

    def _handle_structural_failure(self, attr_string: str, reason: str) -> ValidationResult:
        return ValidationResult(
            state=AttrState.EXCEPTION,
            original_string=attr_string,
            validated_string="ATTR|STRUCTURAL_FAILURE|END",
            slots=[],
            errors=[f"structural_failure:{reason}"],
            repairs=[],
        )
