from __future__ import annotations

import json
import tempfile
from pathlib import Path

from registry.common import build_codeword
from registry.validate import CARACodebook
from validation.validator import AttrState, CARAValidator


def _build_test_validator() -> CARAValidator:
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        pool_a = {"pool_name": "Jazz", "codeword": build_codeword("M", "K4T9X2", "A1")}
        pool_b = {"pool_name": "Techno", "codeword": build_codeword("M", "Q7L3H8", "A1")}
        pool_c = {"pool_name": "Orchestral", "codeword": build_codeword("M", "B2N6R4", "A1")}
        pools_path = root / "pools.json"
        hierarchy_path = root / "hierarchy.json"
        pools_path.write_text(json.dumps([pool_a, pool_b, pool_c]), encoding="utf-8")
        hierarchy = {
            build_codeword("M", "ROOTMS", "A1"): {"level": "root", "name": "Root", "children": []},
            pool_a["codeword"]: {"level": "pool", "name": "Jazz", "parent": build_codeword("M", "ROOTMS", "A1"), "children": []},
            pool_b["codeword"]: {"level": "pool", "name": "Techno", "parent": build_codeword("M", "ROOTMS", "A1"), "children": []},
            pool_c["codeword"]: {"level": "pool", "name": "Orchestral", "parent": build_codeword("M", "ROOTMS", "A1"), "children": []},
        }
        hierarchy_path.write_text(json.dumps(hierarchy), encoding="utf-8")
        codebook = CARACodebook(pools_path, hierarchy_path)
        return CARAValidator(codebook)


def test_state_d_exception() -> None:
    validator = _build_test_validator()
    result = validator.validate("GARBAGE_STRING")
    assert result.state == AttrState.EXCEPTION
    assert result.validated_string is not None
