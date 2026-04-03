"""
AQAA command-line entry point.

Usage::

    python -m aqaa --cycles 5 --output-dir results/
    python -m aqaa --config config.yaml --cycles 10

This module parses CLI arguments, loads configuration, generates
synthetic market data (or loads from file), runs the pipeline, and
saves results to disk.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Dict

import numpy as np

from aqaa.core.config import AQAAConfig, load_config
from aqaa.pipeline import AQAAPipeline

logger = logging.getLogger(__name__)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        prog="aqaa",
        description=(
            "AQAA -- Always-On Autonomous Quant AI Agent.  "
            "Runs the closed-loop alpha generation pipeline."
        ),
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to a YAML configuration file.  If omitted, defaults are used.",
    )
    parser.add_argument(
        "--market",
        type=str,
        default=None,
        help=(
            "Path to a .npz file containing market data arrays "
            "(keys: 'returns', 'predictions', 'dates').  "
            "If omitted, synthetic data is generated."
        ),
    )
    parser.add_argument(
        "--cycles",
        type=int,
        default=1,
        help="Number of 24-hour outer-loop cycles to run (default: 1).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="aqaa_results",
        help="Directory to write results and logs (default: aqaa_results/).",
    )
    return parser.parse_args(argv)


def load_market_data(path: str | None) -> Dict[str, Any]:
    """Load market data from a .npz file or generate synthetic data.

    Parameters
    ----------
    path : str or None
        Path to a NumPy .npz archive.  If None, synthetic data with
        500 trading days and 50 assets is generated.

    Returns
    -------
    dict
        Keys: ``returns``, ``predictions``, ``dates``.
    """
    if path is not None:
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"Market data file not found: {p}")

        logger.info("Loading market data from %s ...", p)
        npz = np.load(str(p), allow_pickle=True)
        data: Dict[str, Any] = {}
        if "returns" in npz:
            data["returns"] = npz["returns"]
        if "predictions" in npz:
            data["predictions"] = npz["predictions"]
        if "dates" in npz:
            data["dates"] = npz["dates"]
        logger.info(
            "Loaded: returns=%s, predictions=%s",
            data.get("returns", np.array([])).shape,
            data.get("predictions", np.array([])).shape,
        )
        return data

    # Generate synthetic market data spanning 2020-2026 so the TIF can
    # partition into train / buffer / test around the LLM cutoff (April 2025).
    # The TIF cutoff is April 2025 with a 6-month buffer, so we need data
    # extending to at least November 2025 for the test partition.
    logger.info("Generating synthetic market data (1500 days x 50 assets) ...")
    rng = np.random.default_rng(42)
    n_assets = 50

    # Build a date array of trading days (Mon-Fri) from 2020-01-02 to 2026-01-02
    all_dates = []
    d = date(2020, 1, 2)
    end_d = date(2026, 3, 1)
    while d <= end_d:
        if d.weekday() < 5:  # Monday=0 ... Friday=4
            all_dates.append(d)
        d += timedelta(days=1)
    dates = np.array(all_dates)
    n_days = len(dates)

    # Simulate correlated asset returns
    daily_mu = 0.0003  # ~7.5% annualised
    daily_vol = 0.015   # ~24% annualised
    returns = rng.normal(daily_mu, daily_vol, size=(n_days, n_assets))

    # Add a common market factor
    market_factor = rng.normal(0, daily_vol * 0.5, size=(n_days, 1))
    returns += market_factor

    # Predictions: noisy version of next-day returns (for backtest)
    noise = rng.normal(0, daily_vol * 2, size=(n_days, n_assets))
    predictions = returns + noise

    return {
        "returns": returns,
        "predictions": predictions,
        "dates": dates,
    }


def save_results(results: Dict[str, Any], output_dir: str) -> Path:
    """Save pipeline results to a JSON file.

    Parameters
    ----------
    results : dict
        Output of ``AQAAPipeline.get_results()``.
    output_dir : str
        Target directory.

    Returns
    -------
    Path
        Path to the written JSON file.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Sanitise for JSON serialisation
    def _default(obj: Any) -> Any:
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (date,)):
            return obj.isoformat()
        return str(obj)

    path = out / "pipeline_results.json"
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2, default=_default)

    logger.info("Results saved to %s", path)
    return path


def main(argv: list[str] | None = None) -> int:
    """Main entry point."""
    args = parse_args(argv)

    # Load configuration
    if args.config:
        logger.info("Loading config from %s", args.config)
        config = load_config(args.config)
    else:
        config = AQAAConfig()

    # Point log directory to the output directory
    config.logging.log_dir = str(Path(args.output_dir) / "logs")

    # Load or generate market data
    market_data = load_market_data(args.market)

    # Instantiate and run the pipeline
    pipeline = AQAAPipeline(config)
    pipeline.run(market_data, n_cycles=args.cycles)

    # Collect and save results
    results = pipeline.get_results()
    save_results(results, args.output_dir)

    # Print summary to stdout
    print("\n" + "=" * 60)
    print("AQAA Pipeline Summary")
    print("=" * 60)
    print(f"  Cycles completed        : {len(results['cycles'])}")
    print(f"  Total hypotheses        : {results['total_hypotheses_generated']}")
    print(f"  Total factors deployed  : {results['total_factors_deployed']}")
    print(f"  Total factors retired   : {results['total_factors_retired']}")
    print(f"  Final portfolio IC      : {results['final_portfolio_ic']:.4f}")
    print(f"  Knowledge base entries  : {results['knowledge_base_stats']['total_factors']}")
    print(f"  Output directory        : {args.output_dir}/")
    print("=" * 60)

    return 0


if __name__ == "__main__":
    sys.exit(main())
