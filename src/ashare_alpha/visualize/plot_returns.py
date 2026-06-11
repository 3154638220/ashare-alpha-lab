import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from ashare_alpha.analysis.metrics import parse_trade_dates


def plot_annual_return(nav, save_path="results/figures/annual_return.png"):
    nav = nav.copy()
    nav["trade_date"] = parse_trade_dates(nav["trade_date"])
    nav["nav"] = nav["total_value"] / nav["total_value"].iloc[0]
    nav["year"] = nav["trade_date"].dt.year

    annual = nav.groupby("year")["nav"].agg(["first", "last"])
    annual["return"] = annual["last"] / annual["first"] - 1
    annual = annual.reset_index()

    colors = ["#d62728" if r < 0 else "#2ca02c" for r in annual["return"]]

    plt.figure(figsize=(10, 5))
    plt.bar(annual["year"].astype(str), annual["return"], color=colors)
    plt.axhline(y=0, color="black", linewidth=0.5)
    plt.title("Annual Return")
    plt.xlabel("Year")
    plt.ylabel("Return")
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_monthly_heatmap(nav, save_path="results/figures/monthly_heatmap.png"):
    nav = nav.copy()
    nav["trade_date"] = parse_trade_dates(nav["trade_date"])
    nav["nav"] = nav["total_value"] / nav["total_value"].iloc[0]
    nav["year"] = nav["trade_date"].dt.year
    nav["month"] = nav["trade_date"].dt.month

    monthly = nav.groupby("year")["nav"].apply(
        lambda x: x.pct_change().fillna(0)
    ).reset_index(level=0, drop=True)

    nav["monthly_ret"] = monthly

    pivot = nav.pivot_table(
        values="monthly_ret",
        index="year",
        columns="month",
        aggfunc="sum",
    )

    plt.figure(figsize=(10, 6))
    plt.imshow(pivot, cmap="RdYlGn", aspect="auto", vmin=-0.15, vmax=0.15)
    plt.colorbar(label="Monthly Return")
    plt.title("Monthly Return Heatmap")
    plt.xlabel("Month")
    plt.ylabel("Year")
    plt.xticks(range(12), range(1, 13))
    plt.yticks(range(len(pivot.index)), pivot.index.astype(str))
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
