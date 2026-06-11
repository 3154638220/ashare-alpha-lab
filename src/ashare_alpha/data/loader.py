from pathlib import Path
import pandas as pd


def load_panel(name: str, data_dir: str = "data/processed") -> pd.DataFrame:
    path = Path(data_dir) / f"{name}.parquet"
    return pd.read_parquet(path)


def load_factor_panel(
    factor_dir: str = "data/factors",
) -> pd.DataFrame:
    path = Path(factor_dir) / "factor_panel.parquet"
    return pd.read_parquet(path)


def load_factor_score(
    factor_dir: str = "data/factors",
) -> pd.DataFrame:
    path = Path(factor_dir) / "factor_score.parquet"
    return pd.read_parquet(path)


def load_target_weights(
    signal_dir: str = "data/signals",
) -> pd.DataFrame:
    path = Path(signal_dir) / "target_weights.parquet"
    return pd.read_parquet(path)


def load_raw(name: str, raw_dir: str = "data/raw") -> pd.DataFrame:
    path = Path(raw_dir) / f"{name}.parquet"
    return pd.read_parquet(path)
