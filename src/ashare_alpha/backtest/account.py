import pandas as pd


class Account:
    def __init__(self, init_cash: float):
        self.cash = init_cash
        self.positions: dict = {}
        self.total_value = init_cash
        self.init_cash = init_cash

    def update_market_value(
        self, trade_date: str, price_df: pd.DataFrame
    ) -> float:
        market_value = 0.0

        for ts_code, shares in self.positions.items():
            row = price_df[
                (price_df["trade_date"] == trade_date)
                & (price_df["ts_code"] == ts_code)
            ]

            if row.empty:
                continue

            close_price = row.iloc[0]["adj_close"]
            market_value += shares * close_price

        self.total_value = self.cash + market_value
        return self.total_value

    def update_market_value_optimized(
        self, trade_date: str, today_price: pd.DataFrame
    ) -> float:
        market_value = 0.0

        for ts_code, shares in self.positions.items():
            if ts_code not in today_price.index:
                continue

            close_price = today_price.loc[ts_code, "adj_close"]
            market_value += shares * close_price

        self.total_value = self.cash + market_value
        return self.total_value
