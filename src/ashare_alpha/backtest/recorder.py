from pathlib import Path
import pandas as pd


class Recorder:
    def __init__(self, result_dir: str = "results"):
        self.result_dir = Path(result_dir)
        self.result_dir.mkdir(parents=True, exist_ok=True)

    def save_nav(self, nav: pd.DataFrame) -> None:
        path = self.result_dir / "nav.csv"
        nav.to_csv(path, index=False)

    def save_trades(self, trades: pd.DataFrame) -> None:
        if trades.empty:
            return
        path = self.result_dir / "trades.csv"
        trades.to_csv(path, index=False)

    def save_positions(self, positions: pd.DataFrame) -> None:
        if positions.empty:
            return
        path = self.result_dir / "positions.csv"
        positions.to_csv(path, index=False)

    def save_all(self, results: dict) -> None:
        if "nav" in results:
            self.save_nav(results["nav"])
        if "trades" in results:
            self.save_trades(results["trades"])
        if "positions" in results:
            self.save_positions(results["positions"])
