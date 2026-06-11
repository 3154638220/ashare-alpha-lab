import pandas as pd


def calc_composite_score(factors: pd.DataFrame, weights: dict) -> pd.DataFrame:
    out = factors.copy()

    out["score"] = 0.0

    for factor_name, weight in weights.items():
        if factor_name in out.columns:
            out["score"] += weight * out[factor_name].fillna(0)

    return out[
        [c for c in [
            "ts_code",
            "trade_date",
            "value",
            "quality",
            "growth",
            "lowvol",
            "momentum",
            "reversal",
            "score",
        ] if c in out.columns]
    ]
