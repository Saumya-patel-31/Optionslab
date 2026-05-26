from __future__ import annotations

import math

from optionlab.models import Greeks, Option, OptionType
from optionlab.pricers.base import Pricer


def _d1(S: float, K: float, T: float, r: float, sigma: float, q: float) -> float:
    return (math.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))


def _d2(d1: float, sigma: float, T: float) -> float:
    return d1 - sigma * math.sqrt(T)


def _phi(x: float) -> float:
    """Standard normal PDF."""
    return math.exp(-0.5 * x**2) / math.sqrt(2 * math.pi)


def _Phi(x: float) -> float:
    """Standard normal CDF via math.erfc for full precision."""
    return 0.5 * math.erfc(-x / math.sqrt(2))


class BlackScholesPricer(Pricer):
    """
    Closed-form Black-Scholes-Merton pricer for European options.

    Supports continuous dividend yield q (Merton 1973 extension).
    All Greeks are computed analytically — no finite differences needed.
    """

    def price(self, option: Option) -> float:
        S, K, T, r, sigma, q = (
            option.S, option.K, option.T,
            option.r, option.sigma, option.q,
        )
        d1 = _d1(S, K, T, r, sigma, q)
        d2 = _d2(d1, sigma, T)

        if option.is_call:
            return (
                S * math.exp(-q * T) * _Phi(d1)
                - K * math.exp(-r * T) * _Phi(d2)
            )
        # Put via direct BS formula (not put-call parity) for numerical clarity
        return (
            K * math.exp(-r * T) * _Phi(-d2)
            - S * math.exp(-q * T) * _Phi(-d1)
        )

    def greeks(self, option: Option) -> Greeks:
        S, K, T, r, sigma, q = (
            option.S, option.K, option.T,
            option.r, option.sigma, option.q,
        )
        d1 = _d1(S, K, T, r, sigma, q)
        d2 = _d2(d1, sigma, T)
        sqrt_T = math.sqrt(T)
        e_qT = math.exp(-q * T)
        e_rT = math.exp(-r * T)
        phi_d1 = _phi(d1)

        # Delta: sensitivity of price to spot
        if option.is_call:
            delta = e_qT * _Phi(d1)
        else:
            delta = -e_qT * _Phi(-d1)

        # Gamma: rate of change of delta (same for calls and puts)
        gamma = e_qT * phi_d1 / (S * sigma * sqrt_T)

        # Theta: time decay per calendar day
        # The -1/365 converts from per-year to per-day
        theta_annual = (
            -S * e_qT * phi_d1 * sigma / (2 * sqrt_T)
            + q * S * e_qT * (_Phi(d1) if option.is_call else -_Phi(-d1))
            - r * K * e_rT * (_Phi(d2) if option.is_call else -_Phi(-d2))
        )
        theta = theta_annual / 365

        # Vega: sensitivity to a 1% absolute change in vol
        vega = S * e_qT * phi_d1 * sqrt_T / 100

        # Rho: sensitivity to a 1% absolute change in rate
        if option.is_call:
            rho = K * T * e_rT * _Phi(d2) / 100
        else:
            rho = -K * T * e_rT * _Phi(-d2) / 100

        return Greeks(delta=delta, gamma=gamma, theta=theta, vega=vega, rho=rho)

    def put_call_parity_check(self, option: Option) -> float:
        """
        Verifies put-call parity: C - P = S*e^(-qT) - K*e^(-rT).
        Returns the residual — should be ~0 for a correct pricer.
        """
        from optionlab.models import OptionType

        call_opt = Option(
            S=option.S, K=option.K, T=option.T,
            r=option.r, sigma=option.sigma,
            option_type=OptionType.CALL, q=option.q,
        )
        put_opt = Option(
            S=option.S, K=option.K, T=option.T,
            r=option.r, sigma=option.sigma,
            option_type=OptionType.PUT, q=option.q,
        )
        C = self.price(call_opt)
        P = self.price(put_opt)
        lhs = C - P
        rhs = option.S * math.exp(-option.q * option.T) - option.K * math.exp(-option.r * option.T)
        return lhs - rhs
