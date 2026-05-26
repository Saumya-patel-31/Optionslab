"""
Tests for Binomial and Monte Carlo pricers.

Key test: both must converge to Black-Scholes for European options,
since BS is the exact analytical solution for GBM.
"""
import pytest

from optionlab import (
    BlackScholesPricer,
    BinomialPricer,
    MonteCarloPricer,
    Option,
    OptionType,
)

bs  = BlackScholesPricer()
crr = BinomialPricer(steps=1000)
mc  = MonteCarloPricer(n_paths=200_000, n_steps=252, seed=0)

ATM_CALL = Option(S=100, K=100, T=1.0, r=0.05, sigma=0.20, option_type=OptionType.CALL)
ATM_PUT  = Option(S=100, K=100, T=1.0, r=0.05, sigma=0.20, option_type=OptionType.PUT)


class TestBinomialConvergence:
    @pytest.mark.parametrize("option", [ATM_CALL, ATM_PUT])
    def test_converges_to_bs(self, option):
        bs_price  = bs.price(option)
        crr_price = crr.price(option)
        assert abs(crr_price - bs_price) < 0.01   # within 1 cent at 1000 steps

    def test_american_put_above_european(self):
        # American put >= European put due to early exercise premium
        european = crr.price(ATM_PUT, american=False)
        american = crr.price(ATM_PUT, american=True)
        assert american >= european - 1e-8   # allow tiny floating point slack

    def test_american_call_no_dividend_equals_european(self):
        # Without dividends, early exercise of a call is never optimal
        european = crr.price(ATM_CALL, american=False)
        american = crr.price(ATM_CALL, american=True)
        assert abs(american - european) < 0.05

    def test_more_steps_more_accurate(self):
        coarse = BinomialPricer(steps=50).price(ATM_CALL)
        fine   = BinomialPricer(steps=500).price(ATM_CALL)
        bs_ref = bs.price(ATM_CALL)
        assert abs(fine - bs_ref) < abs(coarse - bs_ref)


class TestMonteCarloPricer:
    @pytest.mark.parametrize("option", [ATM_CALL, ATM_PUT])
    def test_converges_to_bs(self, option):
        bs_price = bs.price(option)
        mc_price = mc.price(option)
        assert abs(mc_price - bs_price) < 0.15   # MC has sampling noise

    def test_std_error_reasonable(self):
        price, se = mc.price_with_std_error(ATM_CALL)
        assert se > 0
        assert se < 0.5   # std error should be well under 50 cents at 200k paths

    def test_antithetic_reduces_std_error(self):
        mc_plain = MonteCarloPricer(n_paths=50_000, antithetic=False, seed=1)
        mc_anti  = MonteCarloPricer(n_paths=50_000, antithetic=True,  seed=1)
        _, se_plain = mc_plain.price_with_std_error(ATM_CALL)
        _, se_anti  = mc_anti.price_with_std_error(ATM_CALL)
        assert se_anti < se_plain

    def test_price_always_positive(self):
        assert mc.price(ATM_CALL) > 0
        assert mc.price(ATM_PUT) > 0
