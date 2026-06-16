import pandas as pd

from ashare_alpha.logger import logger


def generate_target_weights(
    signal: pd.DataFrame,
    config: dict,
) -> pd.DataFrame:
    portfolio_cfg = config["strategy"]["portfolio"]
    top_n = portfolio_cfg.get("top_n", 50)

    df = signal.copy()

    if df.empty:
        return df

    if "rank" not in df.columns:
        df = df.sort_values("score", ascending=False)
        df["rank"] = range(1, len(df) + 1)

    df = df.head(top_n)

    if df.empty:
        return df

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

    logger.info(
        "Target weights: %d stocks, max single=%.4f, total=%.4f",
        len(df),
        df["target_weight"].max(),
        df["target_weight"].sum(),
    )

    return df
