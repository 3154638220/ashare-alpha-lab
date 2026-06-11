import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot_industry_exposure(exposure, save_path="results/figures/industry_exposure.png"):
    pivot = exposure.pivot(
        index="trade_date",
        columns="industry_name",
        values="weight",
    ).fillna(0)

    top_n = min(10, pivot.shape[1])
    top_industries = pivot.sum().sort_values(ascending=False).head(top_n).index
    pivot_top = pivot[top_industries]

    plt.figure(figsize=(14, 7))
    pivot_top.plot.area(figsize=(14, 7), alpha=0.8)
    plt.title("Industry Exposure (Top 10)")
    plt.xlabel("Date")
    plt.ylabel("Weight")
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
