from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class OptionType(str, Enum):
    CALL = "call"
    PUT = "put"


class ExerciseStyle(str, Enum):
    EUROPEAN = "european"
    AMERICAN = "american"


@dataclass(frozen=True)
class Option:
    """Represents a single vanilla option contract."""

    S: float        # current spot price of the underlying
    K: float        # strike price
    T: float        # time to expiry in years (e.g. 30 days = 30/365)
    r: float        # annualised risk-free rate (e.g. 0.05 = 5%)
    sigma: float    # annualised volatility (e.g. 0.20 = 20%)
    option_type: OptionType = OptionType.CALL
    q: float = 0.0  # continuous dividend yield

    def __post_init__(self) -> None:
        if self.S <= 0:
            raise ValueError(f"Spot price must be positive, got {self.S}")
        if self.K <= 0:
            raise ValueError(f"Strike must be positive, got {self.K}")
        if self.T <= 0:
            raise ValueError(f"Time to expiry must be positive, got {self.T}")
        if self.sigma <= 0:
            raise ValueError(f"Volatility must be positive, got {self.sigma}")

    @property
    def is_call(self) -> bool:
        return self.option_type == OptionType.CALL

    @property
    def intrinsic_value(self) -> float:
        if self.is_call:
            return max(self.S - self.K, 0.0)
        return max(self.K - self.S, 0.0)

    @property
    def moneyness(self) -> float:
        """S / K — above 1 means in-the-money for a call."""
        return self.S / self.K


@dataclass
class Greeks:
    """Container for all first and second order option sensitivities."""

    delta: float = 0.0   # dV/dS
    gamma: float = 0.0   # d²V/dS²
    theta: float = 0.0   # dV/dt  (per calendar day)
    vega: float = 0.0    # dV/dσ  (per 1% move in vol)
    rho: float = 0.0     # dV/dr  (per 1% move in rate)

    def __repr__(self) -> str:
        return (
            f"Greeks(delta={self.delta:.4f}, gamma={self.gamma:.4f}, "
            f"theta={self.theta:.4f}/day, vega={self.vega:.4f}/1%vol, "
            f"rho={self.rho:.4f}/1%rate)"
        )
