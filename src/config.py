import yaml
import os
from typing import Any, Dict

CONFIG_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "config.yaml")

def load_config() -> Dict[str, Any]:
    """Loads the YAML configuration file."""
    if not os.path.exists(CONFIG_PATH):
        raise FileNotFoundError(f"Configuration file not found at {CONFIG_PATH}")
    
    with open(CONFIG_PATH, "r") as f:
        config = yaml.safe_load(f)
    
    return config

# Global config object
try:
    CONFIG = load_config()
except Exception as e:
    print(f"Error loading config: {e}")
    CONFIG = {} 
