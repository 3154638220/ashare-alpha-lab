import pandas as pd

from .factor_ic import calc_rank_ic_matrix, summarize_rank_ic


def calc_factor_decay(
    factor_scores: pd.DataFrame,
    price: pd.DataFrame,
    factor_names: list[str],
    horizons: list[int] | tuple[int, ...] = (5, 10, 20, 40, 60),
) -> pd.DataFrame:
    records = []

    for horizon in horizons:
        ic_all = calc_rank_ic_matrix(
            factor_scores,
            price,
            factor_names=factor_names,
            horizon=int(horizon),
        )

        for factor_name in factor_names:
            ic_df = ic_all[ic_all["factor"] == factor_name][["trade_date", "rank_ic"]].copy()
            summary = summarize_rank_ic(ic_df)
            records.append({
                "factor": factor_name,
                "horizon": int(horizon),
                **summary,
            })

    return pd.DataFrame(records)
