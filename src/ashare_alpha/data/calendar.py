import pandas as pd


def get_trade_dates(cal: pd.DataFrame) -> list[str]:
    return cal.loc[cal["is_open"] == 1, "cal_date"].sort_values().tolist()


def get_first_trading_day_of_month(trade_dates: list[str]) -> list[str]:
    result = []
    for d in trade_dates:
        month = d[:6]
        if not result or result[-1][:6] != month:
            result.append(d)
    return result


def get_prev_trade_date(
    trade_date: str, trade_dates: list[str]
) -> str | None:
    idx = trade_dates.index(trade_date) if trade_date in trade_dates else -1
    if idx > 0:
        return trade_dates[idx - 1]
    return None


def get_next_trade_date(
    trade_date: str, trade_dates: list[str]
) -> str | None:
    idx = trade_dates.index(trade_date) if trade_date in trade_dates else -1
    if 0 <= idx < len(trade_dates) - 1:
        return trade_dates[idx + 1]
    return None
