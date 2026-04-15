from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch

from common.config import load_project_config
from common.env import load_env_file

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PRETRAINED_NAME = "stabilityai/stable-audio-open-small"


def project_paths() -> dict[str, Path]:
    return {
        "project_root": PROJECT_ROOT,
        "data_dir": PROJECT_ROOT / "data",
        "metadata_module": PROJECT_ROOT / "model" / "cara_metadata.py",
        "dataset_config": PROJECT_ROOT / "model" / "dataset_config.json",
        "model_config": PROJECT_ROOT / "model" / "model_config.json",
    }


def build_dataset_config(audio_dir: str | Path | None = None, dataset_id: str = "cara_audio") -> dict[str, Any]:
    paths = project_paths()
    audio_dir_path = Path(audio_dir) if audio_dir is not None else paths["data_dir"]
    return {
        "dataset_type": "audio_dir",
        "datasets": [
            {
                "id": dataset_id,
                "path": str(audio_dir_path.resolve()),
                "custom_metadata_module": str(paths["metadata_module"].resolve()),
            }
        ],
        "random_crop": True,
        "drop_last": True,
    }


def write_dataset_config(output_path: str | Path | None = None, audio_dir: str | Path | None = None) -> Path:
    path = Path(output_path) if output_path is not None else project_paths()["dataset_config"]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(build_dataset_config(audio_dir=audio_dir), indent=2), encoding="utf-8")
    return path


def load_pretrained_model(pretrained_name: str = DEFAULT_PRETRAINED_NAME, device: str | None = None):
    from stable_audio_tools.models.pretrained import get_pretrained_model

    load_env_file()
    config = load_project_config()
    configured_name = config.get("huggingface", {}).get("model_name") or config.get("model", {}).get("pretrained_model") or config.get("pretrained_model")
    effective_name = pretrained_name if pretrained_name != DEFAULT_PRETRAINED_NAME else (configured_name or DEFAULT_PRETRAINED_NAME)
    model, model_config = get_pretrained_model(effective_name)
    resolved_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(resolved_device).eval()
    return model, model_config, resolved_device


def generate_conditioned_audio(
    model,
    prompt: str,
    seconds_total: float | None = None,
    seconds_start: float = 0.0,
    steps: int = 100,
    cfg_scale: float = 6.0,
    seed: int = -1,
    device: str | None = None,
    return_latents: bool = False,
    **sampler_kwargs,
):
    from stable_audio_tools.inference.generation import generate_diffusion_cond

    resolved_device = device or next(model.parameters()).device.type
    sample_size = getattr(model, "sample_size", None)
    if sample_size is None:
        sample_size = getattr(model, "min_input_length", 2097152)
    sample_rate = getattr(model, "sample_rate", 44100)
    if seconds_total is None:
        seconds_total = sample_size / sample_rate
    conditioning = [{"prompt": prompt, "seconds_start": seconds_start, "seconds_total": seconds_total}]
    audio = generate_diffusion_cond(
        model,
        steps=steps,
        cfg_scale=cfg_scale,
        conditioning=conditioning,
        batch_size=1,
        sample_size=sample_size,
        sample_rate=sample_rate,
        seed=seed,
        device=resolved_device,
        return_latents=return_latents,
        **sampler_kwargs,
    )
    return audio


def extract_conditioned_hidden_states(
    model,
    latents,
    prompt: str,
    seconds_total: float | None = None,
    seconds_start: float = 0.0,
    timestep: float = 0.5,
    device: str | None = None,
):
    resolved_device = device or next(model.parameters()).device.type
    sample_size = getattr(model, "sample_size", None)
    sample_rate = getattr(model, "sample_rate", 44100)
    if sample_size is None:
        sample_size = getattr(model, "min_input_length", 2097152)
    if seconds_total is None:
        seconds_total = sample_size / sample_rate

    conditioning = [{"prompt": prompt, "seconds_start": seconds_start, "seconds_total": seconds_total}]
    conditioning_tensors = model.conditioner(conditioning, resolved_device)
    conditioning_inputs = model.get_conditioning_inputs(conditioning_tensors)

    if hasattr(latents, "to"):
        latent_tensor = latents.to(resolved_device)
    else:
        latent_tensor = torch.tensor(latents, device=resolved_device)

    model_dtype = next(model.model.parameters()).dtype
    latent_tensor = latent_tensor.to(model_dtype)
    if latent_tensor.ndim == 2:
        latent_tensor = latent_tensor.unsqueeze(0)

    t = torch.full((latent_tensor.shape[0],), timestep, device=resolved_device, dtype=model_dtype)
    _output, info = model.model(latent_tensor, t, return_info=True, **conditioning_inputs)
    return info.get("hidden_states", [])
