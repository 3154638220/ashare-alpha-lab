import pandas as pd

from ashare_alpha.logger import logger


def apply_industry_constraint(
    weights: pd.DataFrame,
    max_industry_weight: float = 0.20,
) -> pd.DataFrame:
    if "industry_code" not in weights.columns:
        return weights

    df = weights.copy()
    industry_weight = df.groupby("industry_code")["target_weight"].sum()

    for ind in industry_weight.index:
        if industry_weight[ind] > max_industry_weight:
            scale = max_industry_weight / industry_weight[ind]
            mask = df["industry_code"] == ind
            df.loc[mask, "target_weight"] *= scale

    weight_sum = df["target_weight"].sum()
    if weight_sum > 0:
        df["target_weight"] = df["target_weight"] / weight_sum

    return df


def apply_position_constraints(
    weights: pd.DataFrame,
    config: dict,
) -> pd.DataFrame:
    portfolio_cfg = config["strategy"]["portfolio"]

    df = weights.copy()

    if "max_industry_weight" in portfolio_cfg:
        df = apply_industry_constraint(df, portfolio_cfg["max_industry_weight"])

    max_stock = portfolio_cfg.get("max_stock_weight", 0.02)
    df["target_weight"] = df["target_weight"].clip(upper=max_stock)

    min_stock = portfolio_cfg.get("min_stock_weight", 0.0)
    df = df[df["target_weight"] >= min_stock]

    weight_sum = df["target_weight"].sum()
    if weight_sum > 0:
        df["target_weight"] = df["target_weight"] / weight_sum

    logger.info("After constraints: %d stocks, total weight=%.4f", len(df), df["target_weight"].sum())

    return df
