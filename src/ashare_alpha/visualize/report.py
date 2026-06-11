from pathlib import Path

from ashare_alpha.logger import logger

from .plot_nav import plot_nav_curve
from .plot_drawdown import plot_drawdown, plot_excess_drawdown
from .plot_returns import plot_annual_return, plot_monthly_heatmap
from .plot_exposure import plot_industry_exposure


def generate_report(
    nav,
    trades=None,
    positions=None,
    benchmarks=None,
    exposure=None,
    output_dir: str = "results",
    figures_dir: str = "results/figures",
) -> None:
    out = Path(output_dir)
    figs = Path(figures_dir)
    figs.mkdir(parents=True, exist_ok=True)

    logger.info("Generating charts ...")

    plot_nav_curve(nav, benchmarks=benchmarks, save_path=str(figs / "nav_curve.png"))
    plot_drawdown(nav, save_path=str(figs / "drawdown.png"))
    if benchmarks:
        plot_excess_drawdown(nav, benchmarks=benchmarks, save_path=str(figs / "excess_drawdown.png"))
    plot_annual_return(nav, save_path=str(figs / "annual_return.png"))
    plot_monthly_heatmap(nav, save_path=str(figs / "monthly_heatmap.png"))

    if exposure is not None:
        plot_industry_exposure(exposure, save_path=str(figs / "industry_exposure.png"))

    logger.info("Charts saved to %s", figs)
