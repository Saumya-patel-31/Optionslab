"""
Tests for the Black-Scholes pricer.

Reference values computed independently using the standard BS formula
and cross-checked against QuantLib Python bindings.
"""
import math
import pytest

from optionlab import BlackScholesPricer, Option, OptionType

bs = BlackScholesPricer()

# Canonical test case: S=100, K=100, T=1yr, r=5%, sigma=20%, no dividends
ATM_CALL = Option(S=100, K=100, T=1.0, r=0.05, sigma=0.20, option_type=OptionType.CALL)
ATM_PUT  = Option(S=100, K=100, T=1.0, r=0.05, sigma=0.20, option_type=OptionType.PUT)


class TestBlackScholesPrice:
    def test_atm_call_known_value(self):
        # Known reference: ~10.4506
        price = bs.price(ATM_CALL)
        assert abs(price - 10.4506) < 0.001

    def test_atm_put_known_value(self):
        # Known reference: ~5.5735
        price = bs.price(ATM_PUT)
        assert abs(price - 5.5735) < 0.001

    def test_put_call_parity(self):
        # C - P = S - K*e^(-rT) must hold exactly
        residual = bs.put_call_parity_check(ATM_CALL)
        assert abs(residual) < 1e-10

    def test_deep_itm_call_approaches_intrinsic(self):
        # Deep ITM call should be close to S - K*e^(-rT)
        deep_itm = Option(S=200, K=100, T=1.0, r=0.05, sigma=0.20)
        price = bs.price(deep_itm)
        lower_bound = 200 - 100 * math.exp(-0.05)
        assert price > lower_bound

    def test_deep_otm_call_near_zero(self):
        deep_otm = Option(S=50, K=200, T=0.1, r=0.05, sigma=0.20)
        price = bs.price(deep_otm)
        assert price < 0.001

    def test_price_always_positive(self):
        for S in [80, 100, 120]:
            for K in [80, 100, 120]:
                for opt_type in [OptionType.CALL, OptionType.PUT]:
                    opt = Option(S=S, K=K, T=1.0, r=0.05, sigma=0.20, option_type=opt_type)
                    assert bs.price(opt) >= 0.0

    @pytest.mark.parametrize("option_type", [OptionType.CALL, OptionType.PUT])
    def test_price_increases_with_vol(self, option_type):
        low_vol  = Option(S=100, K=100, T=1.0, r=0.05, sigma=0.10, option_type=option_type)
        high_vol = Option(S=100, K=100, T=1.0, r=0.05, sigma=0.40, option_type=option_type)
        assert bs.price(high_vol) > bs.price(low_vol)

    @pytest.mark.parametrize("option_type", [OptionType.CALL, OptionType.PUT])
    def test_price_increases_with_time(self, option_type):
        short = Option(S=100, K=100, T=0.1, r=0.05, sigma=0.20, option_type=option_type)
        long  = Option(S=100, K=100, T=2.0, r=0.05, sigma=0.20, option_type=option_type)
        assert bs.price(long) > bs.price(short)


class TestBlackScholesGreeks:
    def test_call_delta_between_0_and_1(self):
        greeks = bs.greeks(ATM_CALL)
        assert 0.0 < greeks.delta < 1.0

    def test_put_delta_between_minus1_and_0(self):
        greeks = bs.greeks(ATM_PUT)
        assert -1.0 < greeks.delta < 0.0

    def test_atm_call_delta_near_half(self):
        # ATM call delta should be slightly above 0.5
        greeks = bs.greeks(ATM_CALL)
        assert 0.5 < greeks.delta < 0.65

    def test_gamma_positive(self):
        # Gamma is always positive for long options
        assert bs.greeks(ATM_CALL).gamma > 0
        assert bs.greeks(ATM_PUT).gamma > 0

    def test_call_put_gamma_equal(self):
        # Gamma is identical for call and put at same strike (from BS formula)
        call_gamma = bs.greeks(ATM_CALL).gamma
        put_gamma  = bs.greeks(ATM_PUT).gamma
        assert abs(call_gamma - put_gamma) < 1e-10

    def test_theta_negative(self):
        # Long options lose value as time passes (all else equal)
        assert bs.greeks(ATM_CALL).theta < 0
        assert bs.greeks(ATM_PUT).theta < 0

    def test_vega_positive(self):
        assert bs.greeks(ATM_CALL).vega > 0
        assert bs.greeks(ATM_PUT).vega > 0

    def test_call_rho_positive(self):
        assert bs.greeks(ATM_CALL).rho > 0

    def test_put_rho_negative(self):
        assert bs.greeks(ATM_PUT).rho < 0


class TestOptionModel:
    def test_frozen_dataclass_immutable(self):
        with pytest.raises((TypeError, AttributeError)):
            ATM_CALL.S = 200  # type: ignore[misc]

    def test_negative_spot_raises(self):
        with pytest.raises(ValueError):
            Option(S=-1, K=100, T=1.0, r=0.05, sigma=0.20)

    def test_zero_expiry_raises(self):
        with pytest.raises(ValueError):
            Option(S=100, K=100, T=0, r=0.05, sigma=0.20)

    def test_intrinsic_value_call(self):
        itm = Option(S=110, K=100, T=1.0, r=0.05, sigma=0.20)
        assert itm.intrinsic_value == 10.0

    def test_intrinsic_value_otm_call(self):
        otm = Option(S=90, K=100, T=1.0, r=0.05, sigma=0.20)
        assert otm.intrinsic_value == 0.0
