"""
Tests for vol_smile.py.

We never hit the network in tests. Instead we patch yfinance at the
boundary and feed in synthetic options chain data.  This means tests
run offline, fast, and deterministically.
"""

from __future__ import annotations

import math
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from optionlab.vol_smile import (
    _filter_chain,
    _mid_price,
    _time_to_expiry,
    fetch_smile_data,
    list_expiries,
)


# ---------------------------------------------------------------------------
# Unit tests — pure helpers, no mocking needed
# ---------------------------------------------------------------------------

class TestMidPrice:
    def test_uses_midpoint_when_bid_ask_present(self):
        row = pd.Series({"bid": 1.00, "ask": 1.20, "lastPrice": 0.50})
        assert _mid_price(row) == pytest.approx(1.10)

    def test_falls_back_to_last_price_when_no_bid(self):
        row = pd.Series({"bid": 0.0, "ask": 0.0, "lastPrice": 2.50})
        assert _mid_price(row) == pytest.approx(2.50)

    def test_zero_bid_but_positive_ask_uses_last(self):
        # Crossed market edge case — don't trust the spread
        row = pd.Series({"bid": 0.0, "ask": 1.50, "lastPrice": 1.20})
        assert _mid_price(row) == pytest.approx(1.20)

    def test_all_zero_returns_zero(self):
        row = pd.Series({"bid": 0.0, "ask": 0.0, "lastPrice": 0.0})
        assert _mid_price(row) == 0.0


class TestFilterChain:
    def _make_chain(self) -> pd.DataFrame:
        """Synthetic options chain around S=100."""
        return pd.DataFrame({
            "strike"    : [70,  80,  90,  100, 110, 120, 150],
            "bid"       : [0.0, 2.0, 3.0, 5.0, 3.0, 1.5, 0.0],
            "ask"       : [0.0, 2.2, 3.3, 5.5, 3.3, 1.8, 0.1],
            "lastPrice" : [0.0, 2.1, 3.1, 5.2, 3.1, 1.6, 0.05],
        })

    def test_removes_zero_bid_rows(self):
        df = _filter_chain(self._make_chain(), S=100)
        assert (df["bid"] > 0).all()

    def test_removes_far_wing_strikes(self):
        # 70 and 150 are outside ±40% of S=100
        df = _filter_chain(self._make_chain(), S=100, wing_pct=0.40)
        assert 70  not in df["strike"].values
        assert 150 not in df["strike"].values

    def test_keeps_near_atm_strikes(self):
        df = _filter_chain(self._make_chain(), S=100)
        assert 100 in df["strike"].values
        assert 90  in df["strike"].values
        assert 110 in df["strike"].values

    def test_empty_input_returns_empty(self):
        empty = pd.DataFrame(columns=["strike", "bid", "ask", "lastPrice"])
        result = _filter_chain(empty, S=100)
        assert result.empty


class TestTimeToExpiry:
    def test_future_date_returns_positive_years(self):
        import datetime
        future = (datetime.date.today() + datetime.timedelta(days=365)).isoformat()
        T = _time_to_expiry(future)
        assert 0.99 < T < 1.01

    def test_30_days_approximately_correct(self):
        import datetime
        future = (datetime.date.today() + datetime.timedelta(days=30)).isoformat()
        T = _time_to_expiry(future)
        assert abs(T - 30 / 365) < 0.005

    def test_past_date_raises(self):
        with pytest.raises(ValueError, match="past"):
            _time_to_expiry("2020-01-01")


# ---------------------------------------------------------------------------
# Integration-style tests — yfinance fully mocked
# ---------------------------------------------------------------------------

def _make_fake_ticker(spot: float, expiry: str) -> MagicMock:
    """
    Build a mock yfinance Ticker that returns a synthetic options chain.
    The chain uses realistic-looking prices so our IV solver can actually
    recover sensible volatilities.
    """
    from optionlab import BlackScholesPricer, Option, OptionType

    bs = BlackScholesPricer()
    import datetime
    T = (datetime.date.fromisoformat(expiry) - datetime.date.today()).days / 365.0
    r, sigma = 0.05, 0.20

    strikes = [spot * m for m in [0.85, 0.90, 0.95, 1.00, 1.05, 1.10, 1.15]]

    def _chain_df(opt_type: OptionType) -> pd.DataFrame:
        rows = []
        for K in strikes:
            opt   = Option(S=spot, K=K, T=T, r=r, sigma=sigma, option_type=opt_type)
            price = bs.price(opt)
            rows.append({
                "strike"    : K,
                "bid"       : round(price * 0.98, 2),
                "ask"       : round(price * 1.02, 2),
                "lastPrice" : round(price, 2),
            })
        return pd.DataFrame(rows)

    chain = SimpleNamespace(
        calls=_chain_df(OptionType.CALL),
        puts =_chain_df(OptionType.PUT),
    )

    mock_tk = MagicMock()
    mock_tk.fast_info = {"lastPrice": spot}
    mock_tk.option_chain.return_value = chain
    mock_tk.options = [expiry]
    return mock_tk


class TestFetchSmileData:
    @pytest.fixture()
    def expiry(self) -> str:
        import datetime
        return (datetime.date.today() + datetime.timedelta(days=60)).isoformat()

    def test_returns_non_empty_dataframes(self, expiry: str):
        with patch("optionlab.vol_smile.yf") as mock_yf:
            mock_yf.Ticker.return_value = _make_fake_ticker(100.0, expiry)
            calls, puts, S, T, rate = fetch_smile_data("FAKE", expiry, r=0.05)

        assert not calls.empty
        assert not puts.empty

    def test_spot_matches_ticker(self, expiry: str):
        with patch("optionlab.vol_smile.yf") as mock_yf:
            mock_yf.Ticker.return_value = _make_fake_ticker(250.0, expiry)
            _, _, S, _, _ = fetch_smile_data("FAKE", expiry, r=0.05)

        assert S == pytest.approx(250.0)

    def test_implied_vols_close_to_true_vol(self, expiry: str):
        """Round-trip: prices generated at sigma=0.20 → IVs should recover ~0.20."""
        with patch("optionlab.vol_smile.yf") as mock_yf:
            mock_yf.Ticker.return_value = _make_fake_ticker(100.0, expiry)
            calls, puts, S, T, rate = fetch_smile_data("FAKE", expiry, r=0.05)

        for df in [calls, puts]:
            for iv in df["impl_vol"]:
                assert abs(iv - 0.20) < 0.02   # within 2 vol points of truth

    def test_columns_present(self, expiry: str):
        with patch("optionlab.vol_smile.yf") as mock_yf:
            mock_yf.Ticker.return_value = _make_fake_ticker(100.0, expiry)
            calls, puts, S, T, rate = fetch_smile_data("FAKE", expiry, r=0.05)

        for df in [calls, puts]:
            assert "strike"    in df.columns
            assert "mid_price" in df.columns
            assert "impl_vol"  in df.columns

    def test_no_nan_in_impl_vol(self, expiry: str):
        with patch("optionlab.vol_smile.yf") as mock_yf:
            mock_yf.Ticker.return_value = _make_fake_ticker(100.0, expiry)
            calls, puts, _, _, _ = fetch_smile_data("FAKE", expiry, r=0.05)

        assert not calls["impl_vol"].isna().any()
        assert not puts["impl_vol"].isna().any()


class TestListExpiries:
    def test_returns_list_of_strings(self):
        mock_tk = MagicMock()
        mock_tk.options = ("2025-06-20", "2025-07-18", "2025-09-19")
        with patch("optionlab.vol_smile.yf") as mock_yf:
            mock_yf.Ticker.return_value = mock_tk
            result = list_expiries("FAKE")
        assert result == ["2025-06-20", "2025-07-18", "2025-09-19"]
