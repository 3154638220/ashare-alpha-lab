import pandas as pd


def is_suspended(row: pd.Series) -> bool:
    return pd.isna(row["open"]) or pd.isna(row["close"]) or row["vol"] == 0


def is_limit_up(row: pd.Series) -> bool:
    if pd.isna(row.get("up_limit")):
        return False
    return row["open"] >= row["up_limit"]


def is_limit_down(row: pd.Series) -> bool:
    if pd.isna(row.get("down_limit")):
        return False
    return row["open"] <= row["down_limit"]


def can_buy(row: pd.Series) -> bool:
    if is_suspended(row):
        return False
    if is_limit_up(row):
        return False
    return True


def can_sell(row: pd.Series) -> bool:
    if is_suspended(row):
        return False
    if is_limit_down(row):
        return False
    return True


def round_lot_shares(shares: float, lot_size: int = 100) -> int:
    return int(shares // lot_size * lot_size)
