import pandas as pd

from ashare_alpha.logger import logger


def generate_signal(
    score: pd.DataFrame,
    universe: pd.DataFrame,
    config: dict,
) -> pd.DataFrame:
    portfolio_cfg = config["strategy"]["portfolio"]
    top_n = portfolio_cfg.get("top_n", 50)

    valid_codes = universe["ts_code"].unique()
    df = score[score["ts_code"].isin(valid_codes)].copy()

    df = df.sort_values("score", ascending=False)
    df["rank"] = range(1, len(df) + 1)

    df = df.head(top_n)

    weighting = portfolio_cfg.get("weighting", "equal_weight")

    if weighting == "equal_weight":
        df["target_weight"] = 1.0 / len(df)
    else:
        df["target_weight"] = 1.0 / len(df)

    max_stock = portfolio_cfg.get("max_stock_weight", 0.02)
    df["target_weight"] = df["target_weight"].clip(upper=max_stock)

    weight_sum = df["target_weight"].sum()
    if weight_sum > 0:
        df["target_weight"] = df["target_weight"] / weight_sum

    logger.info("Generated signal for %d stocks, sum weight=%.4f", len(df), df["target_weight"].sum())

    return df
