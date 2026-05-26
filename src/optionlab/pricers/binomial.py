from __future__ import annotations

import math

import numpy as np

from optionlab.models import Greeks, Option, OptionType
from optionlab.pricers.base import Pricer


class BinomialPricer(Pricer):
    """
    Cox-Ross-Rubinstein (CRR) binomial tree pricer.

    Works for both European and American options — the key advantage
    over Black-Scholes. At each node we compare the continuation value
    (rolling back the tree) against early exercise value and take the max.

    Convergence to BS improves with more steps; 500-1000 steps is typical
    for production use.
    """

    def __init__(self, steps: int = 500) -> None:
        if steps < 1:
            raise ValueError("steps must be >= 1")
        self.steps = steps

    def price(self, option: Option, american: bool = False) -> float:
        S, K, T, r, sigma, q = (
            option.S, option.K, option.T,
            option.r, option.sigma, option.q,
        )
        n = self.steps
        dt = T / n

        # CRR parameters
        u = math.exp(sigma * math.sqrt(dt))   # up factor
        d = 1.0 / u                            # down factor (ensures recombining tree)
        disc = math.exp(-r * dt)              # one-step discount factor
        # Risk-neutral probability of an up move
        p = (math.exp((r - q) * dt) - d) / (u - d)

        # Terminal stock prices at expiry — vectorised over all 2^n paths
        # The tree recombines, so there are only n+1 distinct terminal nodes
        j = np.arange(n + 1)
        ST = S * (u ** (n - j)) * (d ** j)   # shape: (n+1,)

        # Terminal payoffs
        if option.is_call:
            payoffs = np.maximum(ST - K, 0.0)
        else:
            payoffs = np.maximum(K - ST, 0.0)

        # Backward induction through the tree
        for _ in range(n):
            payoffs = disc * (p * payoffs[:-1] + (1 - p) * payoffs[1:])

            if american:
                # At each interior node, check if early exercise dominates
                ST = ST[:-1] / u   # roll back spot prices one step
                if option.is_call:
                    exercise = np.maximum(ST - K, 0.0)
                else:
                    exercise = np.maximum(K - ST, 0.0)
                payoffs = np.maximum(payoffs, exercise)

        return float(payoffs[0])

    def greeks(self, option: Option, american: bool = False) -> Greeks:
        """
        Greeks via finite differences on the binomial price.
        Less elegant than BS analytics but model-agnostic —
        works for American options where closed-form Greeks don't exist.
        """
        h_S = option.S * 0.01      # 1% bump in spot
        h_sigma = 0.001            # 0.1% bump in vol
        h_r = 0.0001               # 1bp bump in rate
        h_T = 1 / 365             # 1-day bump in time

        def _price(**kwargs) -> float:
            import dataclasses
            return self.price(dataclasses.replace(option, **kwargs), american)

        p0 = _price()
        p_up = _price(S=option.S + h_S)
        p_dn = _price(S=option.S - h_S)
        p_sig_up = _price(sigma=option.sigma + h_sigma)
        p_sig_dn = _price(sigma=option.sigma - h_sigma)
        p_r_up = _price(r=option.r + h_r)
        p_T_dn = _price(T=max(option.T - h_T, 1e-6))

        delta = (p_up - p_dn) / (2 * h_S)
        gamma = (p_up - 2 * p0 + p_dn) / (h_S**2)
        theta = (p_T_dn - p0) / (-h_T)          # per calendar day
        vega = (p_sig_up - p_sig_dn) / (2 * h_sigma * 100)   # per 1% vol
        rho = (p_r_up - p0) / (h_r * 100)        # per 1% rate

        return Greeks(delta=delta, gamma=gamma, theta=theta, vega=vega, rho=rho)
