"""
Tests for the implied volatility solvers.

Core idea: price an option at a known vol, then recover that vol
from the price. The round-trip error should be near machine epsilon.
"""
import math
import pytest

from optionlab import BlackScholesPricer, Option, OptionType
from optionlab.implied_vol import implied_vol_brent, implied_vol_newton, vol_surface

bs = BlackScholesPricer()


def _make_option(S=100, K=100, T=1.0, r=0.05, sigma=0.20, option_type=OptionType.CALL):
    return Option(S=S, K=K, T=T, r=r, sigma=sigma, option_type=option_type)


def _market_price(option: Option) -> float:
    return bs.price(option)


class TestImpliedVolBrent:
    @pytest.mark.parametrize("true_sigma", [0.10, 0.20, 0.30, 0.50, 0.80])
    def test_round_trip_call(self, true_sigma):
        opt = _make_option(sigma=true_sigma)
        px = _market_price(opt)
        recovered = implied_vol_brent(opt, px)
        assert abs(recovered - true_sigma) < 1e-6

    @pytest.mark.parametrize("true_sigma", [0.10, 0.20, 0.40])
    def test_round_trip_put(self, true_sigma):
        opt = _make_option(sigma=true_sigma, option_type=OptionType.PUT)
        px = _market_price(opt)
        recovered = implied_vol_brent(opt, px)
        assert abs(recovered - true_sigma) < 1e-6

    @pytest.mark.parametrize("K", [80, 90, 100, 110, 120])
    def test_across_strikes(self, K):
        true_sigma = 0.25
        opt = _make_option(K=K, sigma=true_sigma)
        px = _market_price(opt)
        recovered = implied_vol_brent(opt, px)
        assert abs(recovered - true_sigma) < 1e-5

    def test_below_intrinsic_raises(self):
        opt = _make_option(S=110, K=100)   # intrinsic = 10
        with pytest.raises(ValueError, match="below intrinsic"):
            implied_vol_brent(opt, market_price=5.0)


class TestImpliedVolNewton:
    @pytest.mark.parametrize("true_sigma", [0.15, 0.25, 0.40])
    def test_round_trip(self, true_sigma):
        opt = _make_option(sigma=true_sigma)
        px = _market_price(opt)
        recovered = implied_vol_newton(opt, px)
        assert abs(recovered - true_sigma) < 1e-6


class TestVolSurface:
    def test_surface_length_matches_input(self):
        true_sigma = 0.20
        strikes = [85, 90, 95, 100, 105, 110, 115]
        options = [_make_option(K=K, sigma=true_sigma) for K in strikes]
        prices = [_market_price(o) for o in options]
        ivs = vol_surface(options, prices)
        assert len(ivs) == len(options)

    def test_surface_recovers_flat_vol(self):
        true_sigma = 0.25
        strikes = [90, 95, 100, 105, 110]
        options = [_make_option(K=K, sigma=true_sigma) for K in strikes]
        prices = [_market_price(o) for o in options]
        ivs = vol_surface(options, prices)
        for iv in ivs:
            assert not math.isnan(iv)
            assert abs(iv - true_sigma) < 1e-5

    def test_bad_price_returns_nan(self):
        opt = _make_option()
        # Price below intrinsic — unsolvable
        ivs = vol_surface([opt], [0.0001])
        assert math.isnan(ivs[0])

    def test_mismatched_lengths_raises(self):
        opts = [_make_option(), _make_option()]
        with pytest.raises(ValueError):
            vol_surface(opts, [5.0])
