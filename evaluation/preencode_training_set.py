from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch


def _to_serialisable(value: Any):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy().tolist()
    if isinstance(value, dict):
        return {key: _to_serialisable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_to_serialisable(item) for item in value]
    return value


def _batch_metadata_to_list(metadata: Any, batch_size: int) -> list[dict[str, Any]]:
    if isinstance(metadata, list):
        if all(isinstance(item, dict) for item in metadata):
            return metadata
        if all(isinstance(item, tuple) and len(item) == 2 for item in metadata):
            merged: list[dict[str, Any]] = [dict() for _ in range(batch_size)]
            for key, values in metadata:
                for idx in range(min(batch_size, len(values))):
                    merged[idx][str(key)] = values[idx]
            return merged
    if isinstance(metadata, tuple):
        return _batch_metadata_to_list(list(metadata), batch_size)
    if isinstance(metadata, dict):
        return [metadata for _ in range(batch_size)]
    return [{} for _ in range(batch_size)]


def _extract_source_id(metadata_item: dict[str, Any], fallback_id: str) -> str:
    for key in ["source_id", "id", "path"]:
        value = metadata_item.get(key)
        if value:
            if key == "path":
                return Path(str(value)).stem
            return str(value)
    return fallback_id


def preencode_dataset(
    dataset_config_path: Path,
    model_config_path: Path | None,
    output_dir: Path,
    pretrained_name: str | None = None,
    ckpt_path: Path | None = None,
    batch_size: int = 1,
    num_workers: int = 0,
    device: str | None = None,
    limit_batches: int | None = None,
) -> dict[str, Any]:
    from stable_audio_tools.data.dataset import create_dataloader_from_config
    from stable_audio_tools.models.factory import create_model_from_config
    from stable_audio_tools.models.pretrained import get_pretrained_model
    from stable_audio_tools.models.utils import copy_state_dict, load_ckpt_state_dict

    with dataset_config_path.open("r", encoding="utf-8") as handle:
        dataset_config = json.load(handle)

    if pretrained_name:
        model, model_config = get_pretrained_model(pretrained_name)
    else:
        if model_config_path is None:
            raise ValueError("model_config_path is required when pretrained_name is not provided")
        with model_config_path.open("r", encoding="utf-8") as handle:
            model_config = json.load(handle)
        model = create_model_from_config(model_config)
        if ckpt_path is not None:
            copy_state_dict(model, load_ckpt_state_dict(str(ckpt_path)))

    resolved_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(resolved_device).eval()

    data_loader = create_dataloader_from_config(
        dataset_config,
        batch_size=batch_size,
        num_workers=num_workers,
        sample_rate=model_config["sample_rate"],
        sample_size=model_config["sample_size"],
        audio_channels=model_config.get("audio_channels", 2),
        shuffle=False,
    )

    latents_dir = output_dir / "latents"
    hidden_dir = output_dir / "dit_hidden"
    meta_dir = output_dir / "preencode_metadata"
    latents_dir.mkdir(parents=True, exist_ok=True)
    hidden_dir.mkdir(parents=True, exist_ok=True)
    meta_dir.mkdir(parents=True, exist_ok=True)

    encoded_count = 0
    hidden_state_mode = "latent_summary_proxy"

    with torch.no_grad():
        for batch_idx, batch in enumerate(data_loader):
            if limit_batches is not None and batch_idx >= limit_batches:
                break
            audio, metadata = batch
            audio = audio.to(resolved_device)
            if audio.ndim == 4 and audio.shape[0] == 1:
                audio = audio[0]
            if model.pretransform is None:
                raise RuntimeError("Loaded model does not expose a pretransform encoder for latent extraction")
            latents = model.pretransform.encode(audio)
            latents_np = latents.detach().cpu().numpy()

            batch_metadata = _batch_metadata_to_list(metadata, latents.shape[0])

            conditioning = []
            for item_index in range(latents.shape[0]):
                md = batch_metadata[item_index]
                conditioning.append(
                    {
                        "prompt": str(md.get("prompt", "high quality audio")),
                        "seconds_start": float(md.get("seconds_start", 0.0)),
                        "seconds_total": float(md.get("seconds_total", model_config["sample_size"] / model_config["sample_rate"])),
                    }
                )

            conditioning_tensors = model.conditioner(conditioning, resolved_device)
            conditioning_inputs = model.get_conditioning_inputs(conditioning_tensors)
            timestep = torch.full((latents.shape[0],), 0.5, device=resolved_device, dtype=latents.dtype)
            _output, info = model.model(latents, timestep, return_info=True, **conditioning_inputs)
            hidden_states = info.get("hidden_states", [])
            selected_layer_indices = sorted(set(idx for idx in [0, len(hidden_states) // 2, len(hidden_states) - 1] if hidden_states))

            for item_index, latent_np in enumerate(latents_np):
                metadata_item = batch_metadata[item_index] if item_index < len(batch_metadata) else {}
                serialised_metadata = _to_serialisable(metadata_item)
                source_id = _extract_source_id(serialised_metadata, f"batch{batch_idx:06d}_{item_index:03d}")
                np.save(latents_dir / f"{source_id}.npy", latent_np)

                if hidden_states:
                    layer_bundle = {}
                    for layer_index in selected_layer_indices:
                        layer_state = hidden_states[layer_index][item_index].detach().cpu().numpy().astype(np.float32)
                        layer_bundle[f"layer_{layer_index}"] = layer_state
                        np.save(hidden_dir / f"{source_id}_layer{layer_index}.npy", layer_state)
                    summary_state = hidden_states[selected_layer_indices[-1]][item_index].detach().cpu().numpy().astype(np.float32)
                    np.save(hidden_dir / f"{source_id}.npy", summary_state)
                    serialised_metadata["dit_hidden_layers"] = list(layer_bundle.keys())
                else:
                    hidden_proxy = latent_np.mean(axis=-1, keepdims=True).astype(np.float32)
                    np.save(hidden_dir / f"{source_id}.npy", hidden_proxy)
                    serialised_metadata["dit_hidden_layers"] = []

                (meta_dir / f"{source_id}.json").write_text(
                    json.dumps(serialised_metadata, indent=2),
                    encoding="utf-8",
                )
                encoded_count += 1

    details = {
        "encoded_files": encoded_count,
        "device": resolved_device,
        "pretrained_name": pretrained_name,
        "hidden_state_mode": "real_transformer_states" if encoded_count > 0 else hidden_state_mode,
        "dataset_config_path": str(dataset_config_path),
        "model_config_path": str(model_config_path) if model_config_path is not None else None,
    }
    (output_dir / "preencode_details.json").write_text(json.dumps(details, indent=2), encoding="utf-8")
    return details


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-config", default="model/dataset_config.json")
    parser.add_argument("--model-config", default=None)
    parser.add_argument("--output-dir", default="data")
    parser.add_argument("--pretrained-name", default="stabilityai/stable-audio-open-small")
    parser.add_argument("--ckpt-path", default=None)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default=None)
    parser.add_argument("--limit-batches", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = preencode_dataset(
        dataset_config_path=Path(args.dataset_config),
        model_config_path=Path(args.model_config) if args.model_config else None,
        output_dir=Path(args.output_dir),
        pretrained_name=args.pretrained_name,
        ckpt_path=Path(args.ckpt_path) if args.ckpt_path else None,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=args.device,
        limit_batches=args.limit_batches,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
