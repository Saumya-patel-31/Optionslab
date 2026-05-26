from __future__ import annotations

from abc import ABC, abstractmethod

from optionlab.models import Greeks, Option


class Pricer(ABC):
    """
    Abstract base for all option pricing models.

    Every concrete pricer must implement price() and greeks().
    This contract lets callers swap Black-Scholes for Monte Carlo
    without changing any surrounding code.
    """

    @abstractmethod
    def price(self, option: Option) -> float:
        """Return the fair value of the option."""
        ...

    @abstractmethod
    def greeks(self, option: Option) -> Greeks:
        """Return all first and second order sensitivities."""
        ...

    def pnl(self, option: Option, premium_paid: float) -> float:
        """P&L at expiry from the perspective of the option buyer."""
        return option.intrinsic_value - premium_paid
