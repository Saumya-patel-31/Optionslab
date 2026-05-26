from __future__ import annotations

import dataclasses
import math

from scipy.optimize import brentq, newton

from optionlab.models import Option
from optionlab.pricers.black_scholes import BlackScholesPricer

_bs = BlackScholesPricer()

# Sensible vol bounds: 0.1% to 1000% annualised
_VOL_LO = 0.001
_VOL_HI = 10.0


def _price_at_vol(option: Option, sigma: float) -> float:
    return _bs.price(dataclasses.replace(option, sigma=sigma))


def implied_vol_brent(
    option: Option,
    market_price: float,
    tol: float = 1e-8,
    max_iter: int = 200,
) -> float:
    """
    Solve for implied vol using Brent's method.

    Brent's method is the gold standard for robust root-finding:
    it combines the guaranteed convergence of bisection with the
    speed of the secant method. It requires no derivative (unlike
    Newton-Raphson) and never diverges when given a valid bracket.

    Raises ValueError if market_price is below intrinsic value
    or if a bracket cannot be found.
    """
    intrinsic = option.intrinsic_value
    if market_price < intrinsic - 1e-8:
        raise ValueError(
            f"Market price {market_price:.4f} is below intrinsic value "
            f"{intrinsic:.4f} — no real implied vol exists."
        )

    objective = lambda sigma: _price_at_vol(option, sigma) - market_price  # noqa: E731

    lo_price = _price_at_vol(option, _VOL_LO) - market_price
    hi_price = _price_at_vol(option, _VOL_HI) - market_price

    if lo_price * hi_price > 0:
        raise ValueError(
            "Could not bracket implied vol. Market price may be outside "
            f"[{_price_at_vol(option, _VOL_LO):.4f}, {_price_at_vol(option, _VOL_HI):.4f}]."
        )

    return brentq(objective, _VOL_LO, _VOL_HI, xtol=tol, maxiter=max_iter)


def implied_vol_newton(
    option: Option,
    market_price: float,
    initial_guess: float | None = None,
    tol: float = 1e-8,
    max_iter: int = 50,
) -> float:
    """
    Solve for implied vol using Newton-Raphson with vega as the derivative.

    Faster than Brent when a good initial guess exists (e.g., from a
    previous solve on a nearby strike), but can diverge for deep OTM
    options where vega is near zero. Prefer Brent for robustness.
    """
    sigma0 = initial_guess or _corrado_miller_guess(option, market_price)

    def objective(sigma: float) -> float:
        opt = dataclasses.replace(option, sigma=sigma)
        price = _bs.price(opt)
        # Vega in dollar terms per unit vol change (not per 1% as in Greeks)
        vega = _bs.greeks(opt).vega * 100
        return price - market_price, vega  # type: ignore[return-value]

    # scipy.optimize.newton accepts (f, x0, fprime) where fprime returns f'
    def f(sigma: float) -> float:
        return _price_at_vol(option, sigma) - market_price

    def fprime(sigma: float) -> float:
        opt = dataclasses.replace(option, sigma=sigma)
        return _bs.greeks(opt).vega * 100   # vega per unit vol

    return newton(f, sigma0, fprime=fprime, tol=tol, maxiter=max_iter)


def _corrado_miller_guess(option: Option, market_price: float) -> float:
    """
    Corrado-Miller (1996) closed-form approximation for initial vol guess.
    Accurate to within a few vol points for near-the-money options,
    which is enough to seed Newton-Raphson.
    """
    S, K, T, r = option.S, option.K, option.T, option.r
    F = S * math.exp(r * T)           # forward price
    x = market_price - (F - K) / 2    # adjusted price
    term = (F - K) ** 2 / (2 * math.pi)
    sigma_approx = (
        math.sqrt(2 * math.pi / T)
        * (x + math.sqrt(max(x**2 - term, 0.0)))
        / F
    )
    return max(sigma_approx, _VOL_LO)


def vol_surface(
    options: list[Option],
    market_prices: list[float],
    method: str = "brent",
) -> list[float]:
    """
    Compute implied vols for a list of options (e.g., an entire options chain).

    Returns a list of implied vols aligned with the input options.
    Failed solves return float('nan') instead of raising, so one bad
    strike doesn't abort the whole surface.
    """
    if len(options) != len(market_prices):
        raise ValueError("options and market_prices must have equal length")

    solver = implied_vol_brent if method == "brent" else implied_vol_newton
    results: list[float] = []

    for opt, px in zip(options, market_prices):
        try:
            results.append(solver(opt, px))
        except (ValueError, RuntimeError):
            results.append(float("nan"))

    return results
