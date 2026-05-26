from __future__ import annotations

import math

import numpy as np

from optionlab.models import Greeks, Option
from optionlab.pricers.base import Pricer

try:
    from numba import njit
    _NUMBA_AVAILABLE = True
except ImportError:
    _NUMBA_AVAILABLE = False
    # Fallback: njit becomes a no-op decorator so the code still runs
    def njit(fn):  # type: ignore[misc]
        return fn


@njit
def _simulate_paths(
    S: float,
    r: float,
    q: float,
    sigma: float,
    T: float,
    n_paths: int,
    n_steps: int,
    seed: int,
) -> np.ndarray:
    """
    Simulate GBM paths using Euler-Maruyama discretisation.

    Compiled by numba when available — typically 20-50x faster than
    pure NumPy for large path counts because numba eliminates Python
    overhead and allocates no intermediate arrays.

    Returns terminal stock prices, shape (n_paths,).
    """
    np.random.seed(seed)
    dt = T / n_steps
    drift = (r - q - 0.5 * sigma**2) * dt
    vol_sqrt_dt = sigma * math.sqrt(dt)

    ST = np.empty(n_paths)
    for i in range(n_paths):
        log_S = math.log(S)
        for _ in range(n_steps):
            z = np.random.standard_normal()
            log_S += drift + vol_sqrt_dt * z
        ST[i] = math.exp(log_S)
    return ST


class MonteCarloPricer(Pricer):
    """
    Monte Carlo pricer for European options under GBM.

    Uses antithetic variates to halve variance at no extra simulation cost:
    for every random path z, we also evaluate -z. The two paths are
    negatively correlated, so their average has much lower variance than
    either alone.

    numba acceleration is applied to the inner path simulation loop when
    numba is installed; falls back to pure Python otherwise.
    """

    def __init__(
        self,
        n_paths: int = 100_000,
        n_steps: int = 252,
        seed: int = 42,
        antithetic: bool = True,
    ) -> None:
        self.n_paths = n_paths
        self.n_steps = n_steps
        self.seed = seed
        self.antithetic = antithetic

    def _terminal_prices(self, option: Option) -> np.ndarray:
        half = self.n_paths // 2 if self.antithetic else self.n_paths

        ST = _simulate_paths(
            option.S, option.r, option.q, option.sigma,
            option.T, half, self.n_steps, self.seed,
        )

        if not self.antithetic:
            return ST

        # Antithetic paths: flip every Gaussian draw
        # Achieved by reflecting log-returns: if ST = S*exp(X), antithetic = S*exp(-X + correction)
        log_ratio = np.log(ST / option.S)
        drift = (option.r - option.q - 0.5 * option.sigma**2) * option.T
        antithetic_log = 2 * drift - log_ratio      # mirrors the noise term
        ST_anti = option.S * np.exp(antithetic_log)

        return np.concatenate([ST, ST_anti])

    def price(self, option: Option) -> float:
        ST = self._terminal_prices(option)

        if option.is_call:
            payoffs = np.maximum(ST - option.K, 0.0)
        else:
            payoffs = np.maximum(option.K - ST, 0.0)

        discount = math.exp(-option.r * option.T)
        return float(discount * payoffs.mean())

    def price_with_std_error(self, option: Option) -> tuple[float, float]:
        """Returns (price, standard_error) — useful for reporting MC confidence."""
        ST = self._terminal_prices(option)

        if option.is_call:
            payoffs = np.maximum(ST - option.K, 0.0)
        else:
            payoffs = np.maximum(option.K - ST, 0.0)

        discount = math.exp(-option.r * option.T)
        price = float(discount * payoffs.mean())
        std_error = float(discount * payoffs.std() / math.sqrt(len(payoffs)))
        return price, std_error

    def greeks(self, option: Option) -> Greeks:
        """Pathwise / finite-difference Greeks from Monte Carlo."""
        import dataclasses

        h_S = option.S * 0.01
        h_sigma = 0.001
        h_r = 0.0001
        h_T = 1 / 365

        def _p(**kwargs) -> float:
            return self.price(dataclasses.replace(option, **kwargs))

        p0 = _p()
        p_up = _p(S=option.S + h_S)
        p_dn = _p(S=option.S - h_S)
        p_sig_up = _p(sigma=option.sigma + h_sigma)
        p_sig_dn = _p(sigma=option.sigma - h_sigma)
        p_r_up = _p(r=option.r + h_r)
        p_T_dn = _p(T=max(option.T - h_T, 1e-6))

        from optionlab.models import Greeks
        return Greeks(
            delta=(p_up - p_dn) / (2 * h_S),
            gamma=(p_up - 2 * p0 + p_dn) / h_S**2,
            theta=(p_T_dn - p0) / (-h_T),
            vega=(p_sig_up - p_sig_dn) / (2 * h_sigma * 100),
            rho=(p_r_up - p0) / (h_r * 100),
        )
