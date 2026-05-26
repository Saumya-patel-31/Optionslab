"""
vol_smile.py
============
Fetches a live options chain from Yahoo Finance, computes implied
volatility for every strike using the library's own solver, and plots
the resulting volatility smile / surface.

The pipeline is:
    yfinance options chain
        → filter illiquid / bad-data rows
        → build Option objects for every strike
        → call vol_surface() to solve for implied vol
        → plot IV (%) vs moneyness (K/S) for calls and puts
"""

from __future__ import annotations

import datetime
import warnings
from typing import Optional

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import yfinance as yf

from optionlab.models import Option, OptionType
from optionlab.implied_vol import vol_surface as _vol_surface


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _fetch_risk_free_rate() -> float:
    """
    Pull the current 3-month US T-bill annualised yield from Yahoo Finance
    (ticker ^IRX, quoted in percent).  Falls back to 5% if the fetch fails.
    """
    try:
        irx = yf.Ticker("^IRX")
        rate = irx.fast_info["lastPrice"] / 100.0
        return rate if 0.001 < rate < 0.30 else 0.05
    except Exception:
        return 0.05


def _time_to_expiry(expiry: str) -> float:
    """'YYYY-MM-DD' → fractional years from today."""
    exp_date = datetime.date.fromisoformat(expiry)
    today    = datetime.date.today()
    days     = (exp_date - today).days
    if days <= 0:
        raise ValueError(f"Expiry {expiry!r} is in the past — pick a future date.")
    return days / 365.0


def _mid_price(row: pd.Series) -> float:
    """
    Best-effort market price for a single option row.
    Prefers (bid + ask) / 2 because last trade may be stale.
    Falls back to lastPrice when bid/ask aren't available.
    """
    bid = float(row.get("bid", 0) or 0)
    ask = float(row.get("ask", 0) or 0)
    if bid > 0 and ask > 0 and ask >= bid:
        return (bid + ask) / 2.0
    last = float(row.get("lastPrice", 0) or 0)
    return last


def _filter_chain(df: pd.DataFrame, S: float, wing_pct: float = 0.40) -> pd.DataFrame:
    """
    Remove rows that would make the IV solver fail or return nonsense.

    Kept rows must:
    - Have a positive bid  (market is making a two-sided quote)
    - Have a mid-price above 0.01  (not penny options — IV solver can't handle them)
    - Lie within ±wing_pct of spot  (extreme wings are illiquid and noisy)
    """
    df = df.copy()

    bid_ok   = df["bid"].fillna(0) > 0
    lo, hi   = S * (1 - wing_pct), S * (1 + wing_pct)
    strike_ok = df["strike"].between(lo, hi)

    df = df[bid_ok & strike_ok].copy()
    df["_mid"] = df.apply(_mid_price, axis=1)
    df = df[df["_mid"] > 0.01]

    return df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def list_expiries(ticker: str) -> list[str]:
    """Return all available expiry dates for a ticker (as 'YYYY-MM-DD' strings)."""
    return list(yf.Ticker(ticker).options)


def fetch_smile_data(
    ticker: str,
    expiry: str,
    r: Optional[float] = None,
) -> tuple[pd.DataFrame, pd.DataFrame, float, float, float]:
    """
    Full pipeline: fetch → filter → solve IV → return clean DataFrames.

    Parameters
    ----------
    ticker  : Yahoo Finance symbol, e.g. "SPY", "AAPL", "TSLA"
    expiry  : Expiry date string "YYYY-MM-DD"  (must be a future date)
    r       : Annualised risk-free rate.  Fetched from ^IRX when None.

    Returns
    -------
    calls_df  : DataFrame[strike, mid_price, impl_vol]  for call options
    puts_df   : DataFrame[strike, mid_price, impl_vol]  for put options
    S         : Current spot price
    T         : Time to expiry in years
    rate      : Risk-free rate used
    """
    tk   = yf.Ticker(ticker)
    S    = float(tk.fast_info["lastPrice"])
    T    = _time_to_expiry(expiry)
    rate = r if r is not None else _fetch_risk_free_rate()

    chain = tk.option_chain(expiry)

    result_dfs: list[pd.DataFrame] = []

    for raw_df, opt_type in [(chain.calls, OptionType.CALL), (chain.puts, OptionType.PUT)]:
        df = _filter_chain(raw_df, S)

        if df.empty:
            result_dfs.append(pd.DataFrame(columns=["strike", "mid_price", "impl_vol"]))
            continue

        mid_prices = df["_mid"].tolist()

        options = [
            Option(
                S=S,
                K=float(row["strike"]),
                T=T,
                r=rate,
                sigma=0.20,           # initial sigma is ignored by solver; just needs to be valid
                option_type=opt_type,
            )
            for _, row in df.iterrows()
        ]

        # Suppress convergence warnings — bad strikes produce nan, not crashes
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ivs = _vol_surface(options, mid_prices)

        out = pd.DataFrame({
            "strike"    : df["strike"].values,
            "mid_price" : mid_prices,
            "impl_vol"  : ivs,
        })

        # Drop failed solves and clearly-wrong vols (< 1% or > 500% annualised)
        out = out.dropna(subset=["impl_vol"])
        out = out[(out["impl_vol"] > 0.01) & (out["impl_vol"] < 5.0)]
        out = out.sort_values("strike").reset_index(drop=True)

        result_dfs.append(out)

    return result_dfs[0], result_dfs[1], S, T, rate


def plot_smile(
    ticker: str,
    expiries: list[str],
    r: Optional[float] = None,
    save_path: Optional[str] = None,
    show: bool = True,
) -> plt.Figure:
    """
    Plot the implied volatility smile across one or more expiry dates.

    X-axis  : Moneyness = K / S  (1.0 = at-the-money)
    Y-axis  : Implied volatility in percent
    Layout  : Side-by-side panels for calls and puts.

    Parameters
    ----------
    ticker     : Yahoo Finance ticker
    expiries   : List of 'YYYY-MM-DD' strings — each becomes one coloured line
    r          : Risk-free rate override; fetched automatically when None
    save_path  : If given, saves the figure (PNG/PDF/SVG) instead of displaying
    show       : Call plt.show() at the end (set False when saving in scripts)

    Returns
    -------
    The matplotlib Figure — so callers can customise it further.
    """
    colors = plt.cm.tab10.colors
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
    fig.suptitle(
        f"Implied Volatility Smile — {ticker.upper()}",
        fontsize=14, fontweight="bold", y=0.98,
    )

    spot_display: Optional[float] = None
    rate_display: Optional[float] = None
    any_data = False

    for i, expiry in enumerate(expiries):
        color = colors[i % len(colors)]

        try:
            print(f"  Fetching {ticker} options for expiry {expiry} …")
            calls, puts, S, T, rate = fetch_smile_data(ticker, expiry, r)
            spot_display = S
            rate_display = rate
        except Exception as exc:
            print(f"  ✗ Skipped {expiry}: {exc}")
            continue

        days  = round(T * 365)
        label = f"{expiry}  ({days}d)"

        for ax, df in zip(axes, [calls, puts]):
            if df.empty:
                continue
            moneyness = df["strike"] / S
            ax.plot(
                moneyness,
                df["impl_vol"] * 100,
                "o-",
                color=color,
                label=label,
                linewidth=1.8,
                markersize=4,
                alpha=0.85,
            )
            any_data = True

    # Styling
    for ax, side in zip(axes, ["Calls", "Puts"]):
        ax.axvline(x=1.0, color="black", linestyle="--", linewidth=1,
                   alpha=0.4, label="ATM  (K = S)")
        ax.set_xlabel("Moneyness  (K / S)", fontsize=11)
        ax.set_title(side, fontsize=12, fontweight="semibold")
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda y, _: f"{y:.0f}%"))
        ax.legend(fontsize=8, framealpha=0.7)
        ax.grid(True, alpha=0.25, linestyle=":")
        ax.set_xlim(0.65, 1.40)

    axes[0].set_ylabel("Implied Volatility", fontsize=11)

    # Footer with spot and rate
    if spot_display and rate_display:
        fig.text(
            0.5, 0.005,
            f"Spot: ${spot_display:.2f}   |   Risk-free rate: {rate_display:.2%}",
            ha="center", fontsize=9, color="#666666",
        )

    plt.tight_layout(rect=[0, 0.03, 1, 0.96])

    if not any_data:
        print("  No data could be plotted — all expiries failed or returned empty chains.")

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saved -> {save_path}")

    if show:
        plt.show()

    return fig


def print_smile_table(ticker: str, expiry: str, r: Optional[float] = None) -> None:
    """
    Print a quick summary table of the vol smile to stdout.
    Useful for verifying data before plotting.
    """
    calls, puts, S, T, rate = fetch_smile_data(ticker, expiry, r)
    days = round(T * 365)

    print(f"\n{'='*60}")
    print(f"  {ticker.upper()}  |  Expiry: {expiry}  ({days} days)  |  Spot: ${S:.2f}")
    print(f"  Risk-free rate: {rate:.2%}")
    print(f"{'='*60}")

    combined = pd.merge(
        calls[["strike", "impl_vol"]].rename(columns={"impl_vol": "call_iv"}),
        puts[["strike",  "impl_vol"]].rename(columns={"impl_vol": "put_iv"}),
        on="strike", how="outer",
    ).sort_values("strike")

    combined["moneyness"]  = (combined["strike"] / S).map("{:.3f}".format)
    combined["call_iv"]    = combined["call_iv"].map(lambda x: f"{x*100:.1f}%" if pd.notna(x) else "—")
    combined["put_iv"]     = combined["put_iv"].map(lambda x: f"{x*100:.1f}%" if pd.notna(x) else "—")

    print(combined[["strike", "moneyness", "call_iv", "put_iv"]].to_string(index=False))
    print()
