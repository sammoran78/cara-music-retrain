from __future__ import annotations

import csv
import json
from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel


app = FastAPI(title="CARA Attribution PoC")
ROOT = Path(__file__).resolve().parents[2]


class ValidateRequest(BaseModel):
    attr_string: str


class GenerateRequest(BaseModel):
    prompt: str


@app.get("/api/data/status")
def data_status():
    data_dir = ROOT / "data"
    counts = {}
    if data_dir.exists():
        counts = {path.name: sum(1 for _ in path.rglob("*") if _.is_file()) for path in data_dir.iterdir() if path.is_dir()}
    coverage_path = ROOT / "data" / "enriched_metadata_coverage.json"
    coverage = json.loads(coverage_path.read_text(encoding="utf-8")) if coverage_path.exists() else {}
    return {"counts": counts, "coverage": coverage}


@app.get("/api/registry/pools")
def registry_pools():
    pools_path = ROOT / "registry" / "pools.json"
    if not pools_path.exists():
        return []
    return json.loads(pools_path.read_text(encoding="utf-8"))


@app.get("/api/registry/hierarchy")
def registry_hierarchy():
    hierarchy_path = ROOT / "registry" / "hierarchy.json"
    if not hierarchy_path.exists():
        return {}
    return json.loads(hierarchy_path.read_text(encoding="utf-8"))


@app.get("/api/registry/codeword/{cw}")
def registry_codeword(cw: str):
    pools = registry_pools()
    for entry in pools:
        if entry.get("codeword") == cw:
            return entry
    raise HTTPException(status_code=404, detail="Codeword not found")


@app.get("/api/training/status")
def training_status():
    checkpoint_path = ROOT / "checkpoints" / "attribution_head_v1.pt"
    return {"checkpoint_exists": checkpoint_path.exists(), "status": "idle"}


@app.post("/api/training/start")
def training_start():
    return {"status": "not_implemented", "message": "Training launch will be wired to stable-audio-tools integration later."}


@app.post("/api/training/stop")
def training_stop():
    return {"status": "not_implemented"}


@app.post("/api/generate")
def generate(request: GenerateRequest):
    return {
        "prompt": request.prompt,
        "audio_url": None,
        "attr_string": "ATTR|PENDING@100|PENDING@00|PENDING@00|END",
        "validation_result": {"state": "exception"},
        "baseline_results": {},
    }


@app.post("/api/validate")
def validate(request: ValidateRequest):
    from registry.validate import CARACodebook
    from validation.validator import CARAValidator

    pools_path = ROOT / "registry" / "pools.json"
    hierarchy_path = ROOT / "registry" / "hierarchy.json"
    if not pools_path.exists() or not hierarchy_path.exists():
        raise HTTPException(status_code=400, detail="Registry not built yet")
    codebook = CARACodebook(pools_path, hierarchy_path)
    validator = CARAValidator(codebook)
    result = validator.validate(request.attr_string)
    return {
        "state": result.state.value,
        "original_string": result.original_string,
        "validated_string": result.validated_string,
        "errors": result.errors,
        "repairs": result.repairs,
    }


@app.get("/api/evaluation/metrics")
def evaluation_metrics():
    metrics_path = ROOT / "evaluation" / "metrics_latest.json"
    if not metrics_path.exists():
        return {}
    return json.loads(metrics_path.read_text(encoding="utf-8"))


@app.post("/api/evaluation/run")
def evaluation_run():
    return {"status": "not_implemented", "message": "Run evaluation from the CLI scaffold for now."}


@app.get("/api/evaluation/comparison")
def evaluation_comparison():
    log_path = ROOT / "evaluation_log.csv"
    if not log_path.exists():
        return []
    with log_path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))[:50]
