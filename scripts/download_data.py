#!/usr/bin/env python3
"""
Data download stubs for real market data sources.

This module provides stub functions for downloading actual market data
from public and commercial sources.  Each function documents the data
requirements, expected schema, and how to obtain API credentials.

The synthetic data generator (``generate_data.py``) produces data in
the same schema as these downloaders, so experiments can run with
either source transparently.

Supported data sources
----------------------

US Equities (S&P 500):
    - Yahoo Finance via ``yfinance`` (free, rate-limited)
    - Qlib (Microsoft open-source quant platform, pre-processed)
    - Polygon.io (commercial, requires API key)

China A-Shares (CSI 500):
    - Qlib CN data (``qlib.contrib.data.handler``)
    - Tushare (Chinese market data, requires registration token)
    - AKShare (open-source Chinese financial data)

Crypto (BTC, ETH, etc.):
    - CCXT (unified API for 100+ exchanges)
    - Binance public API (free, rate-limited)
    - CoinGecko API (free tier available)

Usage (CLI)::

    python scripts/download_data.py --source yahoo --market us --output data/
    python scripts/download_data.py --source qlib --market china --output data/
    python scripts/download_data.py --source ccxt --market crypto --output data/

Usage (Python)::

    from scripts.download_data import download_sp500
    df = download_sp500(start="2020-01-01", end="2025-12-31")
"""

from __future__ import annotations

import argparse
import logging
import os
import warnings
from pathlib import Path
from typing import List, Optional

import pandas as pd

logger = logging.getLogger(__name__)


# ======================================================================
# Expected output schema
# ======================================================================
#
# All download functions should return DataFrames matching these schemas
# so that they are interchangeable with the synthetic data generator.
#
# Equities (US / China):
#   Columns: date, ticker, open, high, low, close, volume, returns
#   Optional: pe_ratio, book_to_market, market_cap_bn, earnings_surprise,
#             transaction_cost_bps
#
# Crypto:
#   Columns: datetime, pair, open, high, low, close, volume, returns
#   Optional: transaction_cost_bps
#


# ======================================================================
# US Equities -- S&P 500
# ======================================================================

def download_sp500(
    start: str = "2020-01-01",
    end: str = "2025-12-31",
    source: str = "yahoo",
    tickers: Optional[List[str]] = None,
    output_dir: Optional[str] = None,
) -> pd.DataFrame:
    """Download S&P 500 constituent OHLCV data.

    This is a **stub** -- it documents the expected interface and data
    requirements but does not perform actual downloads.  To enable real
    downloads, install the appropriate package and implement the body.

    Data sources
    ------------

    **Yahoo Finance** (``yfinance``):
        Install: ``pip install yfinance``
        Usage::

            import yfinance as yf
            data = yf.download("AAPL MSFT GOOG", start="2020-01-01",
                               end="2025-12-31", group_by="ticker")

        Rate limits: ~2,000 requests/hour.  For the full S&P 500 universe
        batch tickers in groups of 50 and add a 1-second delay between
        batches.

        Pros: Free, no API key needed.
        Cons: Rate-limited, adjusted close only, no fundamentals.

    **Qlib** (Microsoft):
        Install: ``pip install pyqlib``
        Data setup::

            python -m qlib.run.get_data qlib_data --target_dir ~/.qlib/qlib_data/us_data --region us

        Usage::

            import qlib
            from qlib.data import D
            qlib.init(provider_uri="~/.qlib/qlib_data/us_data")
            df = D.features(instruments="SP500", fields=["$close", "$volume"],
                            start_time="2020-01-01", end_time="2025-12-31")

        Pros: Pre-processed, aligned, factor-ready.
        Cons: Requires ~10 GB disk space; data may lag behind real-time.

    **Polygon.io** (commercial):
        Install: ``pip install polygon-api-client``
        API key: Register at https://polygon.io (free tier: 5 API calls/min).
        Usage::

            from polygon import RESTClient
            client = RESTClient(api_key=os.environ["POLYGON_API_KEY"])
            aggs = client.get_aggs("AAPL", 1, "day", "2020-01-01", "2025-12-31")

        Pros: High quality, real-time, fundamentals available.
        Cons: Paid for higher rate limits.

    Parameters
    ----------
    start, end : str
        Date range in YYYY-MM-DD format.
    source : str
        Data source: ``"yahoo"``, ``"qlib"``, or ``"polygon"``.
    tickers : list of str, optional
        Specific tickers to download.  If None, downloads all current
        S&P 500 constituents (obtained from Wikipedia).
    output_dir : str, optional
        If provided, save the DataFrame to this directory.

    Returns
    -------
    pd.DataFrame
        Long-format DataFrame with columns:
        ``[date, ticker, open, high, low, close, volume, returns]``.

    Raises
    ------
    NotImplementedError
        Always raised in this stub version.
    """
    raise NotImplementedError(
        f"download_sp500(source='{source}') is a stub. "
        "Install the required package and implement the download logic. "
        "See the docstring for instructions.\n\n"
        "Quick start with Yahoo Finance:\n"
        "  pip install yfinance\n"
        "  import yfinance as yf\n"
        "  data = yf.download('AAPL MSFT', start='2020-01-01')\n\n"
        "For development/testing, use the synthetic data generator instead:\n"
        "  python scripts/generate_data.py --market us --output data/"
    )


# ======================================================================
# China A-Shares -- CSI 500
# ======================================================================

def download_csi500(
    start: str = "2020-01-01",
    end: str = "2025-12-31",
    source: str = "qlib",
    tickers: Optional[List[str]] = None,
    output_dir: Optional[str] = None,
) -> pd.DataFrame:
    """Download CSI 500 constituent OHLCV data.

    This is a **stub**.

    Data sources
    ------------

    **Qlib CN Data** (recommended):
        Install: ``pip install pyqlib``
        Data setup::

            python -m qlib.run.get_data qlib_data --target_dir ~/.qlib/qlib_data/cn_data --region cn

        Usage::

            import qlib
            from qlib.data import D
            qlib.init(provider_uri="~/.qlib/qlib_data/cn_data")
            df = D.features(instruments="CSI500", fields=["$close", "$volume"],
                            start_time="2020-01-01", end_time="2025-12-31")

        Pros: Pre-processed, handles suspension/delisting, factor-ready.
        Cons: ~15 GB disk; data update lag.

    **Tushare** (Chinese market data):
        Install: ``pip install tushare``
        Token: Register at https://tushare.pro and obtain an API token.
        Set environment variable: ``export TUSHARE_TOKEN=your_token``
        Usage::

            import tushare as ts
            pro = ts.pro_api(os.environ["TUSHARE_TOKEN"])
            df = pro.index_weight(index_code="000905.SH")  # CSI 500 constituents
            for code in constituents:
                ohlcv = pro.daily(ts_code=code, start_date="20200101", end_date="20251231")

        Rate limits: 500 calls/min on basic plan, higher on paid plans.
        Pros: Comprehensive Chinese market coverage, fundamentals.
        Cons: Requires registration; free tier has limited fields.

    **AKShare** (open-source alternative):
        Install: ``pip install akshare``
        Usage::

            import akshare as ak
            df = ak.stock_zh_a_hist(symbol="000001", period="daily",
                                     start_date="20200101", end_date="20251231")

        Pros: Free, no registration needed.
        Cons: Less reliable, may break with website changes.

    Parameters
    ----------
    start, end : str
        Date range.
    source : str
        Data source: ``"qlib"``, ``"tushare"``, or ``"akshare"``.
    tickers : list of str, optional
        Specific tickers.  If None, downloads all CSI 500 constituents.
    output_dir : str, optional
        If provided, save to this directory.

    Returns
    -------
    pd.DataFrame
        Long-format DataFrame with columns:
        ``[date, ticker, open, high, low, close, volume, returns]``.

    Raises
    ------
    NotImplementedError
        Always raised in this stub version.
    """
    raise NotImplementedError(
        f"download_csi500(source='{source}') is a stub. "
        "Install the required package and implement the download logic. "
        "See the docstring for instructions.\n\n"
        "Quick start with Qlib:\n"
        "  pip install pyqlib\n"
        "  python -m qlib.run.get_data qlib_data --target_dir ~/.qlib/qlib_data/cn_data --region cn\n\n"
        "For development/testing, use the synthetic data generator instead:\n"
        "  python scripts/generate_data.py --market china --output data/"
    )


# ======================================================================
# Crypto
# ======================================================================

def download_crypto(
    pairs: Optional[List[str]] = None,
    start: str = "2020-01-01",
    end: str = "2025-12-31",
    freq: str = "1h",
    source: str = "ccxt",
    exchange: str = "binance",
    output_dir: Optional[str] = None,
) -> pd.DataFrame:
    """Download cryptocurrency OHLCV data.

    This is a **stub**.

    Data sources
    ------------

    **CCXT** (recommended for multi-exchange support):
        Install: ``pip install ccxt``
        Usage::

            import ccxt
            exchange = ccxt.binance()
            ohlcv = exchange.fetch_ohlcv("BTC/USDT", timeframe="1h",
                                          since=exchange.parse8601("2020-01-01T00:00:00Z"),
                                          limit=1000)
            # Returns: [[timestamp, open, high, low, close, volume], ...]

        Notes:
        - Most exchanges limit to 1000 candles per request.  Paginate
          by advancing the ``since`` parameter.
        - Rate limits vary by exchange (Binance: 1200 req/min).
        - For historical data beyond exchange retention, consider
          dedicated data providers.

        Pros: Unified API for 100+ exchanges, well-maintained.
        Cons: Requires pagination for long histories.

    **Binance Public API** (direct):
        No package needed -- use ``requests`` directly::

            import requests
            url = "https://api.binance.com/api/v3/klines"
            params = {"symbol": "BTCUSDT", "interval": "1h",
                      "startTime": 1577836800000, "limit": 1000}
            response = requests.get(url, params=params)

        Pros: No API key for public endpoints, high rate limits.
        Cons: Binance-only, manual pagination.

    **CoinGecko**:
        Install: ``pip install pycoingecko``
        Usage::

            from pycoingecko import CoinGeckoAPI
            cg = CoinGeckoAPI()
            data = cg.get_coin_market_chart_range_by_id(
                id="bitcoin", vs_currency="usd",
                from_timestamp=1577836800, to_timestamp=1735689600)

        Pros: Free tier, no API key for basic access.
        Cons: Daily granularity only for ranges > 90 days.

    Parameters
    ----------
    pairs : list of str, optional
        Trading pairs (default: ``["BTC/USDT", "ETH/USDT"]``).
    start, end : str
        Date range.
    freq : str
        Candle frequency (``"1h"``, ``"4h"``, ``"1d"``).
    source : str
        Data source: ``"ccxt"``, ``"binance"``, or ``"coingecko"``.
    exchange : str
        Exchange name for CCXT (default: ``"binance"``).
    output_dir : str, optional
        If provided, save to this directory.

    Returns
    -------
    pd.DataFrame
        Long-format DataFrame with columns:
        ``[datetime, pair, open, high, low, close, volume, returns]``.

    Raises
    ------
    NotImplementedError
        Always raised in this stub version.
    """
    if pairs is None:
        pairs = ["BTC/USDT", "ETH/USDT"]

    raise NotImplementedError(
        f"download_crypto(source='{source}', exchange='{exchange}') is a stub. "
        "Install the required package and implement the download logic. "
        "See the docstring for instructions.\n\n"
        "Quick start with CCXT:\n"
        "  pip install ccxt\n"
        "  import ccxt\n"
        "  exchange = ccxt.binance()\n"
        "  ohlcv = exchange.fetch_ohlcv('BTC/USDT', '1h')\n\n"
        "For development/testing, use the synthetic data generator instead:\n"
        "  python scripts/generate_data.py --market crypto --output data/"
    )


# ======================================================================
# Utility: S&P 500 constituent list
# ======================================================================

def get_sp500_tickers() -> List[str]:
    """Fetch current S&P 500 constituent tickers from Wikipedia.

    This is a **stub**.  The implementation should scrape the Wikipedia
    table at https://en.wikipedia.org/wiki/List_of_S%26P_500_companies
    or use a static fallback list.

    Returns
    -------
    list of str
        Ticker symbols (e.g. ``["AAPL", "MSFT", "GOOG", ...]``).

    Raises
    ------
    NotImplementedError
        Always raised in this stub version.
    """
    raise NotImplementedError(
        "get_sp500_tickers() is a stub. "
        "Implement by scraping Wikipedia or using a static list.\n\n"
        "Example with pandas:\n"
        "  table = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')\n"
        "  tickers = table[0]['Symbol'].tolist()"
    )


# ======================================================================
# Utility: data validation
# ======================================================================

def validate_ohlcv(df: pd.DataFrame) -> bool:
    """Validate that a DataFrame conforms to the expected OHLCV schema.

    Checks
    ------
    - Required columns present (date/datetime, ticker/pair, open, high,
      low, close, volume).
    - No negative prices.
    - High >= max(open, close) and Low <= min(open, close) for each row.
    - No all-NaN rows.
    - Dates are monotonically increasing per ticker/pair.

    Parameters
    ----------
    df : pd.DataFrame
        The OHLCV DataFrame to validate.

    Returns
    -------
    bool
        True if all checks pass.

    Raises
    ------
    ValueError
        If any validation check fails, with a descriptive message.
    """
    # Determine if this is equity (date, ticker) or crypto (datetime, pair).
    if "date" in df.columns and "ticker" in df.columns:
        time_col, id_col = "date", "ticker"
    elif "datetime" in df.columns and "pair" in df.columns:
        time_col, id_col = "datetime", "pair"
    else:
        raise ValueError(
            "DataFrame must have (date, ticker) or (datetime, pair) columns. "
            f"Found columns: {list(df.columns)}"
        )

    required = [time_col, id_col, "open", "high", "low", "close", "volume"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # No negative prices.
    for col in ["open", "high", "low", "close"]:
        if (df[col] < 0).any():
            n_neg = (df[col] < 0).sum()
            raise ValueError(f"{n_neg} negative values found in '{col}'.")

    # High >= max(open, close), Low <= min(open, close).
    max_oc = df[["open", "close"]].max(axis=1)
    min_oc = df[["open", "close"]].min(axis=1)
    high_violations = (df["high"] < max_oc - 1e-6).sum()
    low_violations = (df["low"] > min_oc + 1e-6).sum()
    if high_violations > 0:
        warnings.warn(
            f"{high_violations} rows where high < max(open, close). "
            "This may indicate data quality issues.",
            stacklevel=2,
        )
    if low_violations > 0:
        warnings.warn(
            f"{low_violations} rows where low > min(open, close). "
            "This may indicate data quality issues.",
            stacklevel=2,
        )

    # No all-NaN price rows.
    price_cols = ["open", "high", "low", "close"]
    all_nan = df[price_cols].isna().all(axis=1).sum()
    if all_nan > 0:
        raise ValueError(f"{all_nan} rows have all-NaN prices.")

    # Monotonic dates per id.
    for name, group in df.groupby(id_col):
        times = pd.to_datetime(group[time_col])
        if not times.is_monotonic_increasing:
            raise ValueError(
                f"Dates are not monotonically increasing for {id_col}='{name}'."
            )

    logger.info("OHLCV validation passed: %d rows, %d instruments.", len(df), df[id_col].nunique())
    return True


# ======================================================================
# CLI entry point
# ======================================================================

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Download real market data for AQAA experiments. "
            "NOTE: This is currently a stub module -- all download functions "
            "raise NotImplementedError with instructions for setting up "
            "real data sources."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Data source setup:

  US Equities (Yahoo Finance):
    pip install yfinance

  US Equities (Qlib):
    pip install pyqlib
    python -m qlib.run.get_data qlib_data --target_dir ~/.qlib/qlib_data/us_data --region us

  China A-Shares (Qlib):
    pip install pyqlib
    python -m qlib.run.get_data qlib_data --target_dir ~/.qlib/qlib_data/cn_data --region cn

  China A-Shares (Tushare):
    pip install tushare
    export TUSHARE_TOKEN=your_token_here

  Crypto (CCXT):
    pip install ccxt

For development/testing without external data, use the synthetic generator:
  python scripts/generate_data.py --market all --output data/
        """,
    )
    parser.add_argument(
        "--source",
        type=str,
        choices=["yahoo", "qlib", "tushare", "akshare", "ccxt", "binance", "coingecko"],
        default="yahoo",
        help="Data source (default: yahoo).",
    )
    parser.add_argument(
        "--market",
        type=str,
        choices=["us", "china", "crypto"],
        default="us",
        help="Which market to download (default: us).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/",
        help="Output directory (default: data/).",
    )
    parser.add_argument(
        "--start",
        type=str,
        default="2020-01-01",
        help="Start date (default: 2020-01-01).",
    )
    parser.add_argument(
        "--end",
        type=str,
        default="2025-12-31",
        help="End date (default: 2025-12-31).",
    )
    return parser.parse_args()


def main() -> None:
    """CLI entry point for data downloads."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    args = _parse_args()
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        if args.market == "us":
            df = download_sp500(
                start=args.start, end=args.end, source=args.source,
                output_dir=str(output_dir),
            )
        elif args.market == "china":
            df = download_csi500(
                start=args.start, end=args.end, source=args.source,
                output_dir=str(output_dir),
            )
        elif args.market == "crypto":
            df = download_crypto(
                start=args.start, end=args.end, source=args.source,
                output_dir=str(output_dir),
            )
    except NotImplementedError as e:
        logger.error("Download not implemented:\n%s", e)
        logger.info(
            "Use the synthetic data generator for development/testing:\n"
            "  python scripts/generate_data.py --market %s --output %s",
            args.market, args.output,
        )
        return

    logger.info("Download complete. Data saved to %s", output_dir)


if __name__ == "__main__":
    main()
