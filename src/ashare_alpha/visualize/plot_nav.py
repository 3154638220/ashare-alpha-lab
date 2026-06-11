import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ashare_alpha.analysis.metrics import align_nav_with_benchmark


def plot_nav_curve(nav, benchmark=None, benchmarks=None, save_path="results/figures/nav_curve.png"):
    nav = nav.copy()
    nav["nav"] = nav["total_value"] / nav["total_value"].iloc[0]

    plt.figure(figsize=(12, 6))
    plt.plot(nav["trade_date"], nav["nav"], label="Strategy", color="#1f77b4")

    if benchmark is not None:
        benchmarks = {"Benchmark": benchmark}

    if benchmarks:
        colors = ["#d62728", "#2ca02c", "#9467bd", "#ff7f0e"]
        for idx, (name, bench) in enumerate(benchmarks.items()):
            aligned = align_nav_with_benchmark(nav, bench)
            if aligned.empty:
                continue
            plt.plot(
                aligned["trade_date"],
                aligned["benchmark_nav"],
                label=name,
                color=colors[idx % len(colors)],
                alpha=0.75,
            )

    plt.title("Strategy NAV")
    plt.xlabel("Date")
    plt.ylabel("NAV")
    plt.legend()
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
