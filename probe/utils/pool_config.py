"""
pool_config.py

Pool definitions and codeword mappings for CARA attribution.
This should match the actual pool definitions used in training.
"""

# Pool definitions matching our Source-License-Genre structure
POOL_DEFINITIONS = {
    "Freesound-CC0-Electronic": {
        "codeword": "M-AA0001-01",
        "label_id": 0,
        "description": "CC0 licensed electronic music from Freesound",
        "source": "Freesound",
        "license": "CC0",
        "genre": "Electronic",
    },
    "Freesound-CC-BY-Ambient": {
        "codeword": "M-BB0002-01",
        "label_id": 1,
        "description": "CC-BY licensed ambient sounds from Freesound",
        "source": "Freesound",
        "license": "CC-BY",
        "genre": "Ambient",
    },
    "FMA-CC0-Jazz": {
        "codeword": "M-CC0003-01",
        "label_id": 2,
        "description": "CC0 licensed jazz from Free Music Archive",
        "source": "FMA",
        "license": "CC0",
        "genre": "Jazz",
    },
    "FMA-CC-BY-Classical": {
        "codeword": "M-DD0004-01",
        "label_id": 3,
        "description": "CC-BY licensed classical music from Free Music Archive",
        "source": "FMA",
        "license": "CC-BY",
        "genre": "Classical",
    },
}

# Pool hierarchy for degradation state computation
# Maps pools to their broader categories
POOL_HIERARCHY = {
    "Freesound-CC0-Electronic": "Electronic",
    "Freesound-CC-BY-Ambient": "Ambient",
    "FMA-CC0-Jazz": "Jazz",
    "FMA-CC-BY-Classical": "Classical",
}

# Reverse mappings for convenience
CODEWORD_TO_POOL = {v["codeword"]: k for k, v in POOL_DEFINITIONS.items()}
LABEL_ID_TO_POOL = {v["label_id"]: k for k, v in POOL_DEFINITIONS.items()}
POOL_TO_CODEWORD = {k: v["codeword"] for k, v in POOL_DEFINITIONS.items()}
POOL_TO_LABEL_ID = {k: v["label_id"] for k, v in POOL_DEFINITIONS.items()}
