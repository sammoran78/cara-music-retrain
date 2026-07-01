from __future__ import annotations

import importlib.util
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[1] / "data_pipeline" / "12_reconcile_download_manifest.py"
MODULE_SPEC = importlib.util.spec_from_file_location("reconcile_download_manifest", MODULE_PATH)
assert MODULE_SPEC and MODULE_SPEC.loader
RECONCILER = importlib.util.module_from_spec(MODULE_SPEC)
MODULE_SPEC.loader.exec_module(RECONCILER)


def test_floop_does_not_false_positive_into_loop_bucket() -> None:
    primary, secondary, style_tags, summary = RECONCILER._infer_genres_and_summary(
        {
            "name": "Floop.wav",
            "tags": ["abstract", "blow", "blowing", "effect", "strange"],
            "description": "A strange flushing sort of noise.",
        }
    )

    assert primary == "Experimental/Noise"
    assert secondary == "Experimental/Noise"
    assert style_tags == ["noise"]
    assert summary == "Experimental/Noise | noise"


def test_drum_oneshot_uses_broad_style_tags_only() -> None:
    primary, secondary, style_tags, summary = RECONCILER._infer_genres_and_summary(
        {
            "name": "Gui_DRUM_BD_hard.wav",
            "tags": ["acoustic", "drum", "guigui", "mono", "set"],
            "description": "Bass drum hard",
        }
    )

    assert primary == "Percussion/Drums"
    assert secondary == "Acoustic Percussion"
    assert style_tags == ["drum one-shot", "acoustic"]
    assert summary == "Percussion/Drums | drum one-shot, acoustic"
