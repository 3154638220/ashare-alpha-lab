import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ashare_alpha.analysis.metrics import align_nav_with_benchmark


def plot_drawdown(nav, save_path="results/figures/drawdown.png"):
    nav = nav.copy()
    nav["nav"] = nav["total_value"] / nav["total_value"].iloc[0]
    nav["cummax"] = nav["nav"].cummax()
    nav["drawdown"] = nav["nav"] / nav["cummax"] - 1

    plt.figure(figsize=(12, 6))
    plt.fill_between(
        range(len(nav)),
        nav["drawdown"],
        0,
        color="#d62728",
        alpha=0.3,
    )
    plt.plot(range(len(nav)), nav["drawdown"], color="#d62728", linewidth=0.8)
    plt.title("Drawdown")
    plt.xlabel("Trading Days")
    plt.ylabel("Drawdown")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_excess_drawdown(nav, benchmarks, save_path="results/figures/excess_drawdown.png"):
    if not benchmarks:
        return

    plt.figure(figsize=(12, 6))
    colors = ["#d62728", "#2ca02c", "#9467bd", "#ff7f0e"]

    for idx, (name, benchmark) in enumerate(benchmarks.items()):
        aligned = align_nav_with_benchmark(nav, benchmark)
        if aligned.empty:
            continue
        plt.plot(
            aligned["trade_date"],
            aligned["excess_drawdown"],
            label=name,
            color=colors[idx % len(colors)],
            linewidth=1.0,
        )

    plt.title("Excess Drawdown")
    plt.xlabel("Date")
    plt.ylabel("Drawdown")
    plt.legend()
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
