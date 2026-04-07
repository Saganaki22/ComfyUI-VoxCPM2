MODEL_CONFIGS = {
    "VoxCPM2": {
        "repo_id": "openbmb/VoxCPM2",
    },
    "VoxCPM1.5": {
        "repo_id": "openbmb/VoxCPM1.5",
    },
    "VoxCPM-0.5B": {
        "repo_id": "openbmb/VoxCPM-0.5B",
    },
}

AVAILABLE_VOXCPM_MODELS = {
    name: {"type": "official", **cfg}
    for name, cfg in MODEL_CONFIGS.items()
}