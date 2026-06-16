import warnings

import pandas as pd


def normalize_factor_weights(weights: dict, available_columns: list[str]) -> tuple[dict, list[str]]:
    available = {
        factor_name: float(weight)
        for factor_name, weight in weights.items()
        if factor_name in available_columns
    }
    missing = [
        factor_name
        for factor_name in weights
        if factor_name not in available_columns
    ]

    total_weight = sum(available.values())
    if not available or abs(total_weight) < 1e-12:
        return {}, missing

    return {
        factor_name: weight / total_weight
        for factor_name, weight in available.items()
    }, missing


def calc_composite_score(factors: pd.DataFrame, weights: dict) -> pd.DataFrame:
    out = factors.copy()
    effective_weights, missing_factors = normalize_factor_weights(
        weights,
        list(out.columns),
    )

    if missing_factors:
        warnings.warn(
            "Ignored missing factor weights and renormalized available factors: "
            + ", ".join(missing_factors),
            RuntimeWarning,
            stacklevel=2,
        )

    out["score"] = 0.0
    factor_cols = list(effective_weights)

    for factor_name, weight in effective_weights.items():
        out["score"] += weight * out[factor_name].fillna(0)
        out[f"weight_{factor_name}"] = weight

    out["factor_count"] = out[factor_cols].notna().sum(axis=1) if factor_cols else 0

    passthrough_cols = [
        "ts_code",
        "trade_date",
        "industry_code",
        "industry_name",
        "value",
        "quality",
        "growth",
        "lowvol",
        "momentum",
        "reversal",
        "score",
        "factor_count",
    ]
    weight_cols = [f"weight_{factor_name}" for factor_name in factor_cols]

    return out[
        [c for c in passthrough_cols + weight_cols if c in out.columns]
    ]
