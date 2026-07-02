from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ace_step_hybrid_stages import stage_adapter_smoke, stage_full_trainer, stage_planner_probe, stage_prepare_tensors


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")


def test_ace_tensor_prepare_and_planner_probe_preserve_cara_binding(tmp_path: Path) -> None:
    input_root = tmp_path / "input"
    output_root = tmp_path / "output"
    audio_path = input_root / "data" / "freesound" / "freesound_123.wav"
    audio_path.parent.mkdir(parents=True, exist_ok=True)
    audio_path.write_bytes(b"fake wav bytes")
    source_row = {
        "chunk_id": "chunk-a",
        "split": "train",
        "audio_path": "data/freesound/freesound_123.wav",
        "duration_sec": 11.88,
        "description": "bright percussion loop",
        "source_example_id": "source-a",
        "cara_source_pool_id": "CARA:AUD:1:5AJN-QVZH-2MZ7:6Q",
        "cara_pool_id": "CARA:AUD:1:5AJN-QVZH-2MZ7:6Q",
        "cara_pool_index": 7,
        "cara_pool_family": "Percussion",
        "cara_pool_family_index": 2,
    }
    _write_jsonl(input_root / "data" / "cara_pool_manifest_v2.jsonl", [source_row])

    report: dict[str, object] = {}
    stage_prepare_tensors(
        SimpleNamespace(
            input_data=str(input_root),
            output_dir=str(output_root),
            max_rows=0,
            dry_run="false",
        ),
        report,
    )

    assert report["status"] == "passed"
    rows = [json.loads(line) for line in (output_root / "ace_tensor_manifest.jsonl").read_text(encoding="utf-8").splitlines()]
    assert rows[0]["record_id"] == "chunk-a"
    assert rows[0]["cara_pool_id"] == "CARA:AUD:1:5AJN-QVZH-2MZ7:6Q"
    assert rows[0]["cara_pool_index"] == 7
    assert rows[0]["ace_conditioning"]["structured_cara"]["family_index"] == 2
    dataset_json = json.loads((output_root / "dataset.json").read_text(encoding="utf-8"))
    assert dataset_json["metadata"]["format"] == "cara_ace_step_full_json_v1"
    assert dataset_json["metadata"]["tag_position"] == "append"
    assert "samples" in dataset_json
    assert dataset_json["samples"][0]["caption"] == "bright percussion loop"
    assert dataset_json["samples"][0]["custom_tag"] == "CARA:AUD:1:5AJN-QVZH-2MZ7:6Q"
    assert dataset_json["samples"][0]["cara_pool_index"] == 7
    assert dataset_json["samples"][0]["prompt_override"] == "caption"
    assert (output_root / "ace_registry_resolver.json").exists()
    assert (output_root / "sidestep_commands.json").exists()

    planner_report: dict[str, object] = {}
    stage_planner_probe(
        SimpleNamespace(
            ace_tensor_dir=str(output_root),
            output_dir=str(output_root / "planner"),
            tensor_manifest_relative_path="ace_tensor_manifest.jsonl",
            max_rows=0,
            dry_run="false",
        ),
        planner_report,
    )

    assert planner_report["status"] == "passed"
    assert planner_report["planner_survival_exact"] == 1.0
    probe_rows = [
        json.loads(line)
        for line in (output_root / "planner" / "planner_survival_manifest.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert probe_rows[0]["expected_cara_pool_id"] == "CARA:AUD:1:5AJN-QVZH-2MZ7:6Q"
    assert probe_rows[0]["survival_exact"] is True


def test_ace_tensor_prepare_derives_labels_from_raw_source_manifest(tmp_path: Path) -> None:
    input_root = tmp_path / "input"
    output_root = tmp_path / "output"
    audio_path = input_root / "data" / "freesound" / "freesound_321.wav"
    audio_path.parent.mkdir(parents=True, exist_ok=True)
    audio_path.write_bytes(b"fake wav bytes")
    source_row = {
        "chunk_id": "raw-source-a",
        "split": "train",
        "audio_path": "data/freesound/freesound_321.wav",
        "duration_sec": 11.88,
        "description": "tag-withheld field recording texture",
        "source_example_id": "source-raw",
        "cara_source_pool_id": "CARA:AUD:1:BBBB-CCCC-DDDD:EE",
        "cara_pool_family": "Atmosphere/Field",
    }
    _write_jsonl(input_root / "data" / "cara_pool_manifest_v2.jsonl", [source_row])

    report: dict[str, object] = {}
    stage_prepare_tensors(
        SimpleNamespace(
            input_data=str(input_root),
            output_dir=str(output_root),
            max_rows=0,
            dry_run="false",
        ),
        report,
    )

    assert report["status"] == "passed"
    assert report["label_derivation"]["derived_counts"]["cara_pool_id"] == 1
    rows = [json.loads(line) for line in (output_root / "ace_tensor_manifest.jsonl").read_text(encoding="utf-8").splitlines()]
    assert rows[0]["cara_pool_id"] == "CARA:AUD:1:BBBB-CCCC-DDDD:EE"
    assert rows[0]["cara_pool_index"] == 0
    assert rows[0]["cara_pool_family"] == "Atmosphere/Field"
    assert rows[0]["cara_pool_family_index"] == 0


def test_ace_tensor_prepare_rejects_missing_audio_without_failing_dataset(tmp_path: Path) -> None:
    input_root = tmp_path / "input"
    output_root = tmp_path / "output"
    audio_path = input_root / "data" / "freesound" / "freesound_654.wav"
    audio_path.parent.mkdir(parents=True, exist_ok=True)
    audio_path.write_bytes(b"fake wav bytes")
    rows = [
        {
            "chunk_id": "present",
            "split": "train",
            "audio_path": "data/freesound/freesound_654.wav",
            "duration_sec": 11.88,
            "description": "usable source",
            "source_example_id": "source-present",
            "cara_source_pool_id": "CARA:AUD:1:AAAA-BBBB-CCCC:DD",
            "cara_pool_family": "Percussion",
        },
        {
            "chunk_id": "missing",
            "split": "train",
            "audio_path": "data/freesound/not_uploaded.wav",
            "duration_sec": 11.88,
            "description": "missing source",
            "source_example_id": "source-missing",
            "cara_source_pool_id": "CARA:AUD:1:EEEE-FFFF-GGGG:HH",
            "cara_pool_family": "Percussion",
        },
    ]
    _write_jsonl(input_root / "data" / "cara_pool_manifest_v2.jsonl", rows)

    report: dict[str, object] = {}
    stage_prepare_tensors(
        SimpleNamespace(
            input_data=str(input_root),
            output_dir=str(output_root),
            max_rows=0,
            dry_run="false",
        ),
        report,
    )

    assert report["status"] == "passed"
    assert report["tensor_rows"] == 1
    assert report["missing_audio_count"] == 1
    tensor_rows = [json.loads(line) for line in (output_root / "ace_tensor_manifest.jsonl").read_text(encoding="utf-8").splitlines()]
    assert tensor_rows[0]["record_id"] == "present"
    rejected = [json.loads(line) for line in (output_root / "rejected_audio_rows.jsonl").read_text(encoding="utf-8").splitlines()]
    assert rejected[0]["record_id"] == "missing"
    assert rejected[0]["reason"] == "missing_audio"


def test_ace_tensor_prepare_converts_aif_with_ffmpeg(tmp_path: Path) -> None:
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir(parents=True, exist_ok=True)
    fake_ffmpeg = fake_bin / "ffmpeg"
    fake_ffmpeg.write_text(
        "#!/usr/bin/env python3\n"
        "import pathlib, sys\n"
        "pathlib.Path(sys.argv[-1]).parent.mkdir(parents=True, exist_ok=True)\n"
        "pathlib.Path(sys.argv[-1]).write_bytes(b'converted wav bytes')\n",
        encoding="utf-8",
    )
    fake_ffmpeg.chmod(0o755)
    old_path = os.environ.get("PATH", "")
    os.environ["PATH"] = f"{fake_bin}{os.pathsep}{old_path}"
    try:
        input_root = tmp_path / "input"
        output_root = tmp_path / "output"
        audio_path = input_root / "data" / "freesound" / "freesound_456.aif"
        audio_path.parent.mkdir(parents=True, exist_ok=True)
        audio_path.write_bytes(b"fake aif bytes")
        source_row = {
            "chunk_id": "chunk-aif",
            "split": "train",
            "audio_path": "data/freesound/freesound_456.aif",
            "duration_sec": 20.0,
            "description": "warm evolving synth texture",
            "source_example_id": "source-aif",
            "cara_source_pool_id": "CARA:AUD:1:AAAA-BBBB-CCCC:DD",
            "cara_pool_id": "CARA:AUD:1:AAAA-BBBB-CCCC:DD",
            "cara_pool_index": 3,
            "cara_pool_family": "Synth",
            "cara_pool_family_index": 1,
        }
        _write_jsonl(input_root / "data" / "cara_pool_manifest_v2.jsonl", [source_row])

        report: dict[str, object] = {}
        stage_prepare_tensors(
            SimpleNamespace(
                input_data=str(input_root),
                output_dir=str(output_root),
                max_rows=0,
                dry_run="false",
            ),
            report,
        )
    finally:
        os.environ["PATH"] = old_path

    assert report["status"] == "passed"
    assert report["converted_audio_count"] == 1
    assert report["conversion_error_count"] == 0
    rows = [json.loads(line) for line in (output_root / "ace_tensor_manifest.jsonl").read_text(encoding="utf-8").splitlines()]
    assert rows[0]["original_audio_path"].endswith("freesound_456.aif")
    assert rows[0]["audio_path"].endswith(".wav")
    assert rows[0]["sidestep_audio_supported"] is True
    assert Path(rows[0]["audio_path"]).exists()
    dataset_json = json.loads((output_root / "dataset.json").read_text(encoding="utf-8"))
    assert dataset_json["samples"][0]["original_audio_path"].endswith("freesound_456.aif")
    assert dataset_json["samples"][0]["audio_path"].endswith(".wav")


def test_ace_tensor_prepare_rejects_failed_conversion_without_failing_dataset(tmp_path: Path) -> None:
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir(parents=True, exist_ok=True)
    fake_ffmpeg = fake_bin / "ffmpeg"
    fake_ffmpeg.write_text("#!/bin/sh\necho conversion failed >&2\nexit 1\n", encoding="utf-8")
    fake_ffmpeg.chmod(0o755)
    old_path = os.environ.get("PATH", "")
    os.environ["PATH"] = f"{fake_bin}{os.pathsep}{old_path}"
    try:
        input_root = tmp_path / "input"
        output_root = tmp_path / "output"
        wav_path = input_root / "data" / "freesound" / "ok.wav"
        aif_path = input_root / "data" / "freesound" / "bad.aif"
        wav_path.parent.mkdir(parents=True, exist_ok=True)
        wav_path.write_bytes(b"fake wav")
        aif_path.write_bytes(b"fake aif")
        rows = [
            {
                "chunk_id": "ok",
                "split": "train",
                "audio_path": "data/freesound/ok.wav",
                "duration_sec": 11.88,
                "description": "usable source",
                "cara_source_pool_id": "CARA:AUD:1:AAAA-BBBB-CCCC:DD",
                "cara_pool_family": "Percussion",
            },
            {
                "chunk_id": "bad",
                "split": "train",
                "audio_path": "data/freesound/bad.aif",
                "duration_sec": 11.88,
                "description": "failed conversion source",
                "cara_source_pool_id": "CARA:AUD:1:EEEE-FFFF-GGGG:HH",
                "cara_pool_family": "Percussion",
            },
        ]
        _write_jsonl(input_root / "data" / "cara_pool_manifest_v2.jsonl", rows)

        report: dict[str, object] = {}
        stage_prepare_tensors(
            SimpleNamespace(input_data=str(input_root), output_dir=str(output_root), max_rows=0, dry_run="false"),
            report,
        )
    finally:
        os.environ["PATH"] = old_path

    assert report["status"] == "passed"
    assert report["tensor_rows"] == 1
    assert report["conversion_error_count"] == 1
    rejected = [json.loads(line) for line in (output_root / "rejected_audio_rows.jsonl").read_text(encoding="utf-8").splitlines()]
    assert rejected[0]["record_id"] == "bad"
    assert rejected[0]["status"] == "failed"


def test_ace_smoke_fails_fast_on_manifest_missing_required_labels(tmp_path: Path) -> None:
    tensor_dir = tmp_path / "tensors"
    output_dir = tmp_path / "smoke"
    _write_jsonl(
        tensor_dir / "ace_tensor_manifest.jsonl",
        [
            {
                "record_id": "bad-label-row",
                "split": "train",
                "duration_sec": 11.88,
                "prompt": "missing labels",
                "cara_pool_id": "CARA:AUD:1:AAAA-BBBB-CCCC:DD",
                "cara_pool_index": 0,
            }
        ],
    )

    report: dict[str, object] = {}
    try:
        stage_adapter_smoke(
            SimpleNamespace(
                ace_tensor_dir=str(tensor_dir),
                output_dir=str(output_dir),
                tensor_manifest_relative_path="ace_tensor_manifest.jsonl",
                max_rows=0,
                variant="cara_head",
                max_train_rows=10,
                max_eval_rows=10,
                max_steps=1,
                batch_size=1,
                learning_rate=1e-3,
                dry_run="false",
            ),
            report,
        )
    except RuntimeError as exc:
        assert "without required CARA labels" in str(exc)
    else:
        raise AssertionError("stage_adapter_smoke should reject a tensor manifest with missing CARA labels")


def test_ace_full_sidestep_path_writes_trainable_delta_checkpoint(tmp_path: Path) -> None:
    tensor_dir = tmp_path / "tensors"
    output_dir = tmp_path / "full"
    smoke_dir = tmp_path / "smoke"
    checkpoint_dir = tmp_path / "ace_checkpoints"
    sidestep_tensor_dir = tmp_path / "sidestep_tensors"
    checkpoint_dir.mkdir(parents=True)
    sidestep_tensor_dir.mkdir(parents=True)
    smoke_dir.mkdir(parents=True)
    (smoke_dir / "ace_hybrid_smoke_metrics.json").write_text("{}", encoding="utf-8")
    _write_jsonl(
        tensor_dir / "ace_tensor_manifest.jsonl",
        [
            {
                "record_id": "row-a",
                "split": "train",
                "duration_sec": 11.88,
                "prompt": "bright percussion loop",
                "cara_source_pool_id": "CARA:AUD:1:AAAA-BBBB-CCCC:DD",
                "cara_pool_id": "CARA:AUD:1:AAAA-BBBB-CCCC:DD",
                "cara_pool_index": 0,
                "cara_pool_family": "Percussion",
                "cara_pool_family_index": 0,
            }
        ],
    )
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir(parents=True)
    fake_sidestep = fake_bin / "sidestep"
    fake_sidestep.write_text(
        "#!/usr/bin/env python3\n"
        "import pathlib, sys\n"
        "out = pathlib.Path(sys.argv[sys.argv.index('--output-dir') + 1])\n"
        "out.mkdir(parents=True, exist_ok=True)\n"
        "(out / 'adapter_model.safetensors').write_bytes(b'fake lora delta')\n"
        "(out / 'adapter_config.json').write_text('{\"adapter\":\"lora\"}', encoding='utf-8')\n",
        encoding="utf-8",
    )
    fake_sidestep.chmod(0o755)
    old_path = os.environ.get("PATH", "")
    os.environ["PATH"] = f"{fake_bin}{os.pathsep}{old_path}"
    try:
        report: dict[str, object] = {}
        stage_full_trainer(
            SimpleNamespace(
                ace_tensor_dir=str(tensor_dir),
                smoke_dir=str(smoke_dir),
                output_dir=str(output_dir),
                tensor_manifest_relative_path="ace_tensor_manifest.jsonl",
                max_rows=0,
                checkpoint="ACE-Step/Ace-Step1.5",
                planner_checkpoint="ACE-Step/acestep-5Hz-lm-0.6B",
                dit_variant="base_or_sft_dit",
                variant="cara_strong",
                checkpoint_dir=str(checkpoint_dir),
                sidestep_tensor_dir=str(sidestep_tensor_dir),
                run_sidestep="true",
                model_variant="base",
                adapter_type="lora",
                rank=64,
                alpha=128,
                batch_size=4,
                learning_rate=1e-4,
                max_steps=10,
                save_every=50,
                num_workers=0,
                timestep_mode="continuous",
                dry_run="false",
            ),
            report,
        )
    finally:
        os.environ["PATH"] = old_path

    assert report["status"] == "passed"
    assert report["checkpoint_strategy"] == "mounted_output_trainable_delta_only"
    checkpoint_path = output_dir / "checkpoints" / "trainable_delta.pt"
    assert checkpoint_path.exists()
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    assert payload["format"] == "cara_ace_trainable_delta_v1"
    assert payload["delta_type"] == "sidestep_lora_adapter_delta"
    assert payload["run_sidestep"] is True
    assert payload["adapter_artifacts"]
    assert payload["adapter_artifacts"][0]["relative_path"].startswith("sidestep_adapter/")


def test_ace_full_sidestep_success_without_adapter_artifact_fails(tmp_path: Path) -> None:
    tensor_dir = tmp_path / "tensors"
    output_dir = tmp_path / "full"
    smoke_dir = tmp_path / "smoke"
    checkpoint_dir = tmp_path / "ace_checkpoints"
    sidestep_tensor_dir = tmp_path / "sidestep_tensors"
    checkpoint_dir.mkdir(parents=True)
    sidestep_tensor_dir.mkdir(parents=True)
    smoke_dir.mkdir(parents=True)
    (smoke_dir / "ace_hybrid_smoke_metrics.json").write_text("{}", encoding="utf-8")
    _write_jsonl(
        tensor_dir / "ace_tensor_manifest.jsonl",
        [
            {
                "record_id": "row-a",
                "split": "train",
                "duration_sec": 11.88,
                "prompt": "bright percussion loop",
                "cara_source_pool_id": "CARA:AUD:1:AAAA-BBBB-CCCC:DD",
                "cara_pool_id": "CARA:AUD:1:AAAA-BBBB-CCCC:DD",
                "cara_pool_index": 0,
                "cara_pool_family": "Percussion",
                "cara_pool_family_index": 0,
            }
        ],
    )
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir(parents=True)
    fake_sidestep = fake_bin / "sidestep"
    fake_sidestep.write_text(
        "#!/usr/bin/env python3\n"
        "import pathlib, sys\n"
        "out = pathlib.Path(sys.argv[sys.argv.index('--output-dir') + 1])\n"
        "out.mkdir(parents=True, exist_ok=True)\n",
        encoding="utf-8",
    )
    fake_sidestep.chmod(0o755)
    old_path = os.environ.get("PATH", "")
    os.environ["PATH"] = f"{fake_bin}{os.pathsep}{old_path}"
    try:
        report: dict[str, object] = {}
        stage_full_trainer(
            SimpleNamespace(
                ace_tensor_dir=str(tensor_dir),
                smoke_dir=str(smoke_dir),
                output_dir=str(output_dir),
                tensor_manifest_relative_path="ace_tensor_manifest.jsonl",
                max_rows=0,
                checkpoint="ACE-Step/Ace-Step1.5",
                planner_checkpoint="ACE-Step/acestep-5Hz-lm-0.6B",
                dit_variant="base_or_sft_dit",
                variant="cara_strong",
                checkpoint_dir=str(checkpoint_dir),
                sidestep_tensor_dir=str(sidestep_tensor_dir),
                run_sidestep="true",
                model_variant="base",
                adapter_type="lora",
                rank=64,
                alpha=128,
                batch_size=4,
                learning_rate=1e-4,
                max_steps=10,
                save_every=50,
                num_workers=0,
                timestep_mode="continuous",
                dry_run="false",
            ),
            report,
        )
    finally:
        os.environ["PATH"] = old_path

    assert report["status"] == "failed"
    assert report["sidestep_result"]["error"] == "Side-Step returned success but no adapter/delta artifact files were found."
    assert not (output_dir / "checkpoints" / "trainable_delta.pt").exists()
