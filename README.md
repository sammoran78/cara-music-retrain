# CARA Attribution Proof-of-Concept

This repository implements a scaffolded version of the CARA attribution proof-of-concept described in `implement-plan.md`.

## Current State

- `implement-plan.md` is the canonical specification
- `PLAN.md` is the execution tracker
- Data pipeline, registry, validation, evaluation, and GUI layers are scaffolded for subset-first development

## Recommended Workflow

1. Build/test against a small subset or synthetic data
2. Generate registry codewords and hierarchy
3. Run metadata enrichment, genre mapping, pool assignment, and soft target construction
4. Generate sidecars and the master registry
5. Add real model integration with `stable-audio-tools`

## Notes

Several components are intentionally scaffold-level and avoid hard runtime coupling to external services or heavyweight model dependencies until those are ready.

## Credentials

- Store local credentials in `.env`
- Supported keys: `HF_TOKEN`, `FREESOUND_CLIENT_ID`, `FREESOUND_CLIENT_SECRET`, `FREESOUND_ACCESS_TOKEN`, `FREESOUND_REFRESH_TOKEN`
- `.env` is ignored by git and loaded by the project configuration helpers

## Data Direction

- Current default assumption is original-quality Freesound downloads rather than preview MP3s
