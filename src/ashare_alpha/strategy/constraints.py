import pandas as pd

from ashare_alpha.logger import logger

TOLERANCE = 1e-10


def _normalize_initial_weights(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["target_weight"] = df["target_weight"].clip(lower=0)
    weight_sum = df["target_weight"].sum()
    if weight_sum > 0:
        df["target_weight"] = df["target_weight"] / weight_sum
    return df


def _scale_industries_to_cap(
    df: pd.DataFrame,
    max_industry_weight: float | None,
) -> pd.DataFrame:
    if max_industry_weight is None:
        return df

    if "industry_code" not in df.columns:
        return df

    df = df.copy()
    industry_weight = df.groupby("industry_code", dropna=False)["target_weight"].sum()

    for industry_code, weight in industry_weight.items():
        if weight > max_industry_weight + TOLERANCE:
            scale = max_industry_weight / weight
            mask = df["industry_code"] == industry_code
            if pd.isna(industry_code):
                mask = df["industry_code"].isna()
            df.loc[mask, "target_weight"] *= scale

    return df


def _redistribute_residual(
    df: pd.DataFrame,
    residual: float,
    max_stock_weight: float | None = None,
    max_industry_weight: float | None = None,
) -> pd.DataFrame:
    if residual <= TOLERANCE or df.empty:
        return df

    df = df.copy()

    for _ in range(100):
        if residual <= TOLERANCE:
            break

        capacity = pd.Series(float("inf"), index=df.index)

        if max_stock_weight is not None:
            capacity = max_stock_weight - df["target_weight"]

        if max_industry_weight is not None and "industry_code" in df.columns:
            industry_weight = df.groupby("industry_code", dropna=False)["target_weight"].transform("sum")
            industry_capacity = max_industry_weight - industry_weight
            capacity = pd.concat([capacity, industry_capacity], axis=1).min(axis=1)

        capacity = capacity.clip(lower=0)
        eligible = capacity > TOLERANCE
        if not eligible.any():
            break

        basis = df.loc[eligible, "_base_weight"].clip(lower=0)
        if basis.sum() <= TOLERANCE:
            basis = pd.Series(1.0, index=basis.index)

        allocation = residual * basis / basis.sum()
        allocation = pd.concat([allocation, capacity.loc[eligible]], axis=1).min(axis=1)

        if max_industry_weight is not None and "industry_code" in df.columns:
            for industry_code, index in df.loc[eligible].groupby("industry_code", dropna=False).groups.items():
                industry_allocation = allocation.loc[index].sum()
                if pd.isna(industry_code):
                    industry_mask = df["industry_code"].isna()
                else:
                    industry_mask = df["industry_code"] == industry_code
                current_industry_weight = df.loc[industry_mask, "target_weight"].sum()
                room = max_industry_weight - current_industry_weight
                if industry_allocation > room + TOLERANCE:
                    scale = max(room, 0.0) / industry_allocation
                    allocation.loc[index] *= scale

        added = allocation.sum()
        if added <= TOLERANCE:
            break

        df.loc[allocation.index, "target_weight"] += allocation
        residual -= added

    return df


def apply_industry_constraint(
    weights: pd.DataFrame,
    max_industry_weight: float | None = 0.20,
) -> pd.DataFrame:
    if max_industry_weight is None:
        return weights

    if "industry_code" not in weights.columns:
        return weights

    df = _normalize_initial_weights(weights)
    df["_base_weight"] = df["target_weight"]

    df = _scale_industries_to_cap(df, max_industry_weight)
    residual = 1.0 - df["target_weight"].sum()
    df = _redistribute_residual(
        df,
        residual,
        max_industry_weight=max_industry_weight,
    )

    residual = 1.0 - df["target_weight"].sum()
    if residual > TOLERANCE:
        logger.warning(
            "Industry constraint left %.4f unallocated because industry caps are binding",
            residual,
        )

    df = df.drop(columns=["_base_weight"])

    return df


def apply_position_constraints(
    weights: pd.DataFrame,
    config: dict,
) -> pd.DataFrame:
    portfolio_cfg = config["strategy"]["portfolio"]

    df = weights.copy()
    if df.empty:
        return df

    df = _normalize_initial_weights(df)
    df["_base_weight"] = df["target_weight"]

    max_stock = portfolio_cfg.get("max_stock_weight")
    if max_stock is not None:
        df["target_weight"] = df["target_weight"].clip(upper=max_stock)

    max_industry = portfolio_cfg.get("max_industry_weight")
    if max_industry is not None:
        df = _scale_industries_to_cap(df, max_industry)

    residual = 1.0 - df["target_weight"].sum()
    df = _redistribute_residual(
        df,
        residual,
        max_stock_weight=max_stock,
        max_industry_weight=max_industry,
    )

    min_stock = portfolio_cfg.get("min_stock_weight", 0.0)
    df = df[df["target_weight"] >= min_stock]

    residual = 1.0 - df["target_weight"].sum()
    df = _redistribute_residual(
        df,
        residual,
        max_stock_weight=max_stock,
        max_industry_weight=max_industry,
    )

    residual = 1.0 - df["target_weight"].sum()
    if residual > TOLERANCE:
        logger.warning(
            "Position constraints left %.4f unallocated because caps are binding",
            residual,
        )

    df = df.drop(columns=["_base_weight"])

    logger.info("After constraints: %d stocks, total weight=%.4f", len(df), df["target_weight"].sum())

    return df
