import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))


def test_derive_ic_weighted_static_weights_excludes_weak_and_negative():
    import pandas as pd
    from ashare_alpha.analysis.weight_experiments import derive_ic_weighted_static_weights

    summary = pd.DataFrame({
        "factor": ["value", "quality", "growth", "momentum", "reversal"],
        "mean_ic": [0.03, 0.003, 0.01, -0.02, 0.05],
    })

    weights = derive_ic_weighted_static_weights(summary, min_mean_ic=0.01)

    assert set(weights) == {"value", "growth", "reversal"}
    assert abs(sum(weights.values()) - 1.0) < 1e-12
    assert weights["reversal"] > weights["value"] > weights["growth"]


def test_portfolio_factor_exposure_is_weighted_by_target_weight():
    import pandas as pd
    from ashare_alpha.analysis.weight_experiments import (
        calc_portfolio_factor_exposure,
        summarize_portfolio_factor_exposure,
    )

    target_weights = pd.DataFrame({
        "rebalance_date": ["20200102", "20200102", "20200203"],
        "ts_code": ["A", "B", "A"],
        "target_weight": [0.25, 0.75, 1.0],
        "industry_code": ["I1", "I2", "I1"],
        "value": [1.0, 3.0, 5.0],
        "reversal": [2.0, 0.0, 4.0],
    })

    exposure = calc_portfolio_factor_exposure(target_weights, ["value", "reversal"])

    first = exposure[exposure["rebalance_date"] == "20200102"].iloc[0]
    assert first["exposure_value"] == 2.5
    assert first["exposure_reversal"] == 0.5
    assert first["max_industry_weight"] == 0.75

    summary = summarize_portfolio_factor_exposure(exposure)
    assert summary["avg_holding_count"] == 1.5
    assert summary["max_observed_industry_weight"] == 1.0
