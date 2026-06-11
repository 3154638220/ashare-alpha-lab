from pathlib import Path
import yaml


def load_yaml(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_config(config_dir: str = "config") -> dict:
    config_dir = Path(config_dir)

    config = {}

    for name in ["data", "strategy", "cost", "backtest"]:
        path = config_dir / f"{name}.yaml"
        config.update(load_yaml(path))

    return config
