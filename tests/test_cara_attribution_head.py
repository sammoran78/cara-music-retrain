from __future__ import annotations

import pytest
import torch
import sys
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cara_attribution_head import (
    CaraAttributionHead,
    attach_cara_attribution_to_training_wrapper,
    attribution_loss_and_metrics,
    build_cara_registry_resolver,
    labels_from_metadata,
    masked_pool_logits,
    validate_cara_manifest_labels,
)
from prepare_model_datasets import _pool_codeword
from smoke_stable_audio_trainer import (
    _create_stable_audio_dataloader,
    _ensure_cara_conditioner,
    _metadata_prompt,
    _patch_loaded_model_cara_conditioners,
    _prune_checkpoint_dir,
    _preview_cara_conditioner_outputs,
)


def _rows() -> list[dict[str, object]]:
    return [
        {
            "chunk_id": "a",
            "split": "train",
            "cara_pool_id": "CARA:AUD:1:0001",
            "cara_pool_index": 0,
            "cara_pool_family": "Percussion",
            "cara_pool_family_index": 0,
        },
        {
            "chunk_id": "b",
            "split": "validation",
            "cara_pool_id": "CARA:AUD:1:0002",
            "cara_pool_index": 1,
            "cara_pool_family": "Ambient",
            "cara_pool_family_index": 1,
        },
    ]


def test_validate_cara_manifest_labels_fails_fast_on_missing_labels() -> None:
    rows = _rows()
    rows[1].pop("cara_pool_index")

    with pytest.raises(ValueError, match="missing CARA label fields"):
        validate_cara_manifest_labels(rows)


def test_build_registry_resolver_maps_pool_family_indices() -> None:
    resolver = build_cara_registry_resolver(_rows())

    assert resolver["pool_count"] == 2
    assert resolver["family_count"] == 2
    assert resolver["pool_by_index"]["0"] == "CARA:AUD:1:0001"
    assert resolver["pool_to_family_index"]["1"] == 1
    assert resolver["registry_hash"]


def test_masked_pool_logits_blocks_pools_outside_target_family() -> None:
    resolver = build_cara_registry_resolver(_rows())
    logits = torch.tensor([[1.0, 10.0]])
    masked = masked_pool_logits(
        logits,
        torch.tensor([0]),
        {int(key): int(value) for key, value in resolver["pool_to_family_index"].items()},
    )

    assert masked[0, 0] == pytest.approx(1.0)
    assert masked[0, 1] < -1e20


def test_labels_from_metadata_supports_list_metadata() -> None:
    labels = labels_from_metadata(
        [
            {"cara_pool_id": "CARA:AUD:1:0001", "cara_pool_index": 0, "cara_pool_family_index": 0},
            {"cara_pool_id": "CARA:AUD:1:0002", "cara_pool_index": 1, "cara_pool_family_index": 1},
        ],
        device=torch.device("cpu"),
    )

    assert labels["pool_index"].tolist() == [0, 1]
    assert labels["family_index"].tolist() == [0, 1]
    assert labels["pool_id"] == ["CARA:AUD:1:0001", "CARA:AUD:1:0002"]


def test_attribution_head_forward_loss_and_registry_decode() -> None:
    torch.manual_seed(7)
    resolver = build_cara_registry_resolver(_rows())
    head = CaraAttributionHead(num_pools=2, num_families=2, hidden_dim=8)
    features = torch.randn(2, 6)
    outputs = head(features)
    labels = {
        "pool_index": torch.tensor([0, 1]),
        "family_index": torch.tensor([0, 1]),
        "pool_id": ["CARA:AUD:1:0001", "CARA:AUD:1:0002"],
    }

    result = attribution_loss_and_metrics(outputs, labels, resolver)

    assert result.loss.ndim == 0
    assert "cara/pool_top1" in result.metrics
    assert len(result.decoded["top1_cara_pool_id"]) == 2
    assert result.decoded["registry_valid"] is True


def test_frozen_cara_head_uses_head_only_optimizer() -> None:
    class DummyWrapper(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.backbone = torch.nn.Linear(2, 2)

        def training_step(self, batch, batch_idx):
            return self.backbone(batch[0]).sum()

        def configure_optimizers(self):
            params = [param for param in self.backbone.parameters() if param.requires_grad]
            return torch.optim.AdamW(params, lr=1e-5)

    model = torch.nn.Linear(2, 2)
    wrapper = DummyWrapper()
    report: dict[str, object] = {}

    attach_cara_attribution_to_training_wrapper(
        wrapper,
        model,
        resolver=build_cara_registry_resolver(_rows()),
        variant="cara_head",
        loss_weight=0.05,
        optimizer_lr=1e-5,
        detach_features=True,
        freeze_backbone=True,
        report=report,
    )
    optimizer = wrapper.configure_optimizers()

    assert isinstance(optimizer, torch.optim.AdamW)
    assert report["cara_attribution"]["optimizer_mode"] == "head_only"
    assert sum(len(group["params"]) for group in optimizer.param_groups) > 0


def test_checkpoint_pruning_keeps_last_and_newest_periodic(tmp_path: Path) -> None:
    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir()
    old = checkpoint_dir / "epoch=0-step=1000.ckpt"
    new = checkpoint_dir / "epoch=0-step=2000.ckpt"
    last = checkpoint_dir / "last.ckpt"
    old.write_text("old", encoding="utf-8")
    new.write_text("new", encoding="utf-8")
    last.write_text("last", encoding="utf-8")

    old_time = 1_700_000_000
    new_time = old_time + 10
    old.touch()
    new.touch()
    last.touch()
    import os

    os.utime(old, (old_time, old_time))
    os.utime(new, (new_time, new_time))

    summary = _prune_checkpoint_dir(checkpoint_dir, keep_last_n=1)

    assert not old.exists()
    assert new.exists()
    assert last.exists()
    assert summary["kept_periodic"] == [new.name]
    assert summary["deleted"] == [old.name]
    assert summary["last_checkpoint_present"] is True


def test_pool_codeword_derives_from_registered_cara_id() -> None:
    assert _pool_codeword({"cara_pool_id": "CARA:AUD:1:5AJN-QVZH-2MZ7:6Q"}) == "5AJN-QVZH-2MZ7"
    assert _pool_codeword({"cara_pool_codeword": "EXPLICIT", "cara_pool_id": "CARA:AUD:1:OTHER:6Q"}) == "EXPLICIT"


def test_cara_strong_keeps_prompt_ordinary_and_adds_native_conditioners() -> None:
    row = {
        "prompt": "rain texture",
        "cara_pool_id": "CARA:AUD:1:5AJN-QVZH-2MZ7:6Q",
        "cara_pool_family": "Ambient",
    }
    assert _metadata_prompt(row, "cara_lite").endswith("CARA_POOL: CARA:AUD:1:5AJN-QVZH-2MZ7:6Q")
    assert _metadata_prompt(row, "cara_strong") == "rain texture"

    model_config = {
        "model": {
            "conditioning": {
                "cond_dim": 768,
                "configs": [{"id": "prompt", "type": "t5", "config": {"max_length": 77}}],
            },
            "diffusion": {
                "config": {"cond_token_dim": 768},
                "cross_attention_cond_ids": ["prompt"],
            },
        }
    }
    report: dict[str, object] = {}

    _ensure_cara_conditioner(model_config, build_cara_registry_resolver(_rows()), report, enabled=True)

    conditioning_configs = model_config["model"]["conditioning"]["configs"]
    conditioner_ids = {item["id"] for item in conditioning_configs}
    assert {"cara_pool_index", "cara_pool_family_index"} <= conditioner_ids
    assert model_config["model"]["diffusion"]["cross_attention_cond_ids"] == [
        "prompt",
        "cara_pool_index",
        "cara_pool_family_index",
    ]
    assert report["cara_native_conditioning"]["enabled"] is True


def test_loaded_model_cara_conditioners_are_patched_and_previewed(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeIntConditioner(torch.nn.Module):
        def __init__(self, output_dim: int, min_val: int = 0, max_val: int = 512) -> None:
            super().__init__()
            self.output_dim = output_dim
            self.min_val = min_val
            self.max_val = max_val

        def forward(self, ints: list[int], device: torch.device | str | None = None):
            clamped = [min(max(int(value), self.min_val), self.max_val) for value in ints]
            tensor = torch.tensor(clamped, device=device).float().view(len(clamped), 1, 1).expand(-1, 1, self.output_dim)
            return tensor, torch.ones(len(clamped), 1, device=device)

    fake_conditioners_module = types.ModuleType("stable_audio_tools.models.conditioners")
    fake_conditioners_module.IntConditioner = FakeIntConditioner
    monkeypatch.setitem(sys.modules, "stable_audio_tools", types.ModuleType("stable_audio_tools"))
    monkeypatch.setitem(sys.modules, "stable_audio_tools.models", types.ModuleType("stable_audio_tools.models"))
    monkeypatch.setitem(sys.modules, "stable_audio_tools.models.conditioners", fake_conditioners_module)

    class DummyModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.anchor = torch.nn.Parameter(torch.zeros(()))
            self.conditioner = types.SimpleNamespace(conditioners=torch.nn.ModuleDict({"prompt": torch.nn.Identity()}))

    model = DummyModel()
    model_config = {"model": {"conditioning": {"cond_dim": 4}}}
    resolver = build_cara_registry_resolver(_rows())
    report: dict[str, object] = {"cara_native_conditioning": {"enabled": True}}

    _patch_loaded_model_cara_conditioners(model, model_config, resolver, report)
    _preview_cara_conditioner_outputs(
        model,
        [{"cara_pool_index": 1, "cara_pool_family_index": 1}],
        resolver,
        report,
    )

    assert {"cara_pool_index", "cara_pool_family_index"} <= set(model.conditioner.conditioners.keys())
    assert report["cara_native_conditioning"]["loaded_model_conditioner_patched"] is True
    assert report["first_batch_cara_conditioner_preview"]["cara_pool_index"]["tensor_shape"] == [1, 1, 4]


def test_zero_worker_stable_audio_dataloader_disables_persistent_workers() -> None:
    def fake_factory(*, num_workers: int):
        dataset = torch.utils.data.TensorDataset(torch.arange(4))
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=2,
            num_workers=num_workers,
            persistent_workers=True,
        )

    loader = _create_stable_audio_dataloader(fake_factory, num_workers=0)

    assert loader.num_workers == 0
    assert loader.persistent_workers is False
