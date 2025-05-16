import os
import yaml

def load_config(config_path):
    """Load and return YAML config."""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    required_keys = ["benchmark_name", "source", "executable", "make_targets", "parameters"]
    for key in required_keys:
        if key not in config:
            raise KeyError(f"Missing required key in config: {key}")

    return config
