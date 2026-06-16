import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))


def test_generate_signal_backfills_industry_from_universe():
    import pandas as pd
    from ashare_alpha.strategy.signal import generate_signal

    score = pd.DataFrame({
        "ts_code": ["A", "B", "C"],
        "trade_date": ["20200102"] * 3,
        "score": [3.0, 2.0, 1.0],
    })
    universe = pd.DataFrame({
        "ts_code": ["A", "B", "C"],
        "industry_code": ["I1", "I2", "I3"],
        "industry_name": ["Ind 1", "Ind 2", "Ind 3"],
    })
    config = {
        "strategy": {
            "portfolio": {
                "top_n": 2,
                "weighting": "equal_weight",
                "max_stock_weight": 0.6,
            }
        }
    }

    result = generate_signal(score, universe, config)

    assert list(result["ts_code"]) == ["A", "B"]
    assert "industry_code" in result.columns
    assert "industry_name" in result.columns
    assert list(result["industry_code"]) == ["I1", "I2"]


def test_apply_industry_constraint_preserves_cap_after_redistribution():
    import pandas as pd
    from ashare_alpha.strategy.constraints import apply_industry_constraint

    weights = pd.DataFrame({
        "ts_code": ["A", "B", "C", "D"],
        "industry_code": ["I1", "I1", "I2", "I2"],
        "target_weight": [0.40, 0.30, 0.20, 0.10],
    })

    result = apply_industry_constraint(weights, max_industry_weight=0.60)
    industry_weight = result.groupby("industry_code")["target_weight"].sum()

    assert abs(result["target_weight"].sum() - 1.0) < 1e-10
    assert industry_weight.max() <= 0.60 + 1e-10


def test_apply_position_constraints_keeps_industry_cap_after_residual_allocation():
    import pandas as pd
    from ashare_alpha.strategy.constraints import apply_position_constraints

    weights = pd.DataFrame({
        "ts_code": [f"S{i}" for i in range(10)],
        "industry_code": ["I1"] * 5 + ["I2"] * 3 + ["I3"] * 2,
        "target_weight": [0.10] * 10,
    })
    config = {
        "strategy": {
            "portfolio": {
                "max_stock_weight": 0.20,
                "max_industry_weight": 0.40,
                "min_stock_weight": 0.0,
            }
        }
    }

    result = apply_position_constraints(weights, config)
    industry_weight = result.groupby("industry_code")["target_weight"].sum()

    assert abs(result["target_weight"].sum() - 1.0) < 1e-10
    assert industry_weight.max() <= 0.40 + 1e-10
    assert result["target_weight"].max() <= 0.20 + 1e-10


def test_apply_position_constraints_allows_no_industry_cap():
    import pandas as pd
    from ashare_alpha.strategy.constraints import apply_position_constraints

    weights = pd.DataFrame({
        "ts_code": [f"S{i}" for i in range(5)],
        "industry_code": ["I1"] * 5,
        "target_weight": [0.20] * 5,
    })
    config = {
        "strategy": {
            "portfolio": {
                "max_stock_weight": 0.50,
                "max_industry_weight": None,
                "min_stock_weight": 0.0,
            }
        }
    }

    result = apply_position_constraints(weights, config)

    assert abs(result["target_weight"].sum() - 1.0) < 1e-10
    assert result.groupby("industry_code")["target_weight"].sum().iloc[0] == 1.0
