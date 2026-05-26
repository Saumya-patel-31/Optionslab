"""
examples/plot_smile.py
======================
Runnable demo — fetches a live SPY options chain and plots the vol smile.

Usage
-----
    # From the project root:
    python examples/plot_smile.py

    # Different ticker:
    python examples/plot_smile.py --ticker AAPL

    # Save to file instead of showing:
    python examples/plot_smile.py --save smile.png
"""

from __future__ import annotations

import argparse
import sys

# Make sure the src layout is importable when running from project root
sys.path.insert(0, "src")

from optionlab.vol_smile import list_expiries, plot_smile, print_smile_table


def pick_expiries(ticker: str, count: int = 3) -> list[str]:
    """Pick the first `count` expiries that are at least 7 days away."""
    import datetime
    cutoff = datetime.date.today() + datetime.timedelta(days=7)
    all_exp = list_expiries(ticker)
    future  = [e for e in all_exp if datetime.date.fromisoformat(e) > cutoff]
    return future[:count]


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot implied vol smile from live options data.")
    parser.add_argument("--ticker", default="SPY",  help="Yahoo Finance ticker (default: SPY)")
    parser.add_argument("--save",   default=None,   help="Save figure to this path instead of showing")
    parser.add_argument("--table",  action="store_true", help="Also print a text summary table")
    args = parser.parse_args()

    ticker = args.ticker.upper()
    print(f"\nFetching available expiries for {ticker} …")

    expiries = pick_expiries(ticker, count=3)
    if not expiries:
        print("No future expiries found — try a different ticker.")
        sys.exit(1)

    print(f"Using expiries: {expiries}\n")

    if args.table:
        # Print a text summary of the nearest expiry
        print_smile_table(ticker, expiries[0])

    plot_smile(
        ticker    = ticker,
        expiries  = expiries,
        save_path = args.save,
        show      = args.save is None,   # don't call plt.show() when saving
    )

    if args.save:
        print(f"\nDone. Figure saved to: {args.save}")


if __name__ == "__main__":
    main()
