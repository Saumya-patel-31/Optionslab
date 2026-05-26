# optionlab

A from-scratch options pricing engine and interactive analytics dashboard built in Python.

Implements three independent pricers — **Black-Scholes-Merton** (analytical), **Binomial CRR tree** (numerical), and **Monte Carlo GBM** (simulation) — and validates convergence across all three. Includes analytical Greeks, multi-leg strategy analysis, and a live implied volatility surface solver.

![Python](https://img.shields.io/badge/Python-3.11+-3B82F6?style=flat-square&logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.35+-EF4444?style=flat-square&logo=streamlit&logoColor=white)

---

## Features

### Pricers
| Model | Method | Complexity |
|---|---|---|
| Black-Scholes-Merton | Closed-form analytical solution | O(1) |
| Binomial (CRR) | Backward induction on recombining lattice | O(n²) |
| Monte Carlo | GBM simulation with antithetic variates | O(n · m) |

### Greeks (analytical, BSM)
All five Greeks derived from first principles — no finite differences.

| Greek | Measures | Formula |
|---|---|---|
| Delta Δ | dV / dS | e^(−qT) · Φ(d₁) |
| Gamma Γ | d²V / dS² | e^(−qT) · φ(d₁) / (S σ √T) |
| Theta Θ | dV / dt (per day) | −[S e^(−qT) φ(d₁) σ / 2√T − …] / 365 |
| Vega ν | dV / dσ (per 1%) | S e^(−qT) φ(d₁) √T / 100 |
| Rho ρ | dV / dr (per 1%) | K T e^(−rT) Φ(d₂) / 100 |

### Interactive Dashboard (Streamlit)

**6 tabs:**

- **Payoff** — P&L diagram for single options or multi-leg strategies (Bull/Bear Spread, Straddle, Strangle, Iron Condor, Butterfly). Breakeven detection, max profit/loss, P&L scenario matrix across spot × time.
- **Greeks** — All-Greek dashboard or single-Greek deep-dive vs. spot. Explanation cards.
- **Heatmap** — Any metric (Price, Δ, Γ, Θ, ν) across the full spot × volatility space. Interactive slice at current vol.
- **Chain** — Theoretical option chain at all strikes around ATM. Put-call parity verification per strike.
- **Vol Smile** — Live options chain from Yahoo Finance. Implied vol solved at every strike using Brent's root-finding method. Smile plot + ATM vol term structure.
- **Models** — Convergence charts (price and absolute error) for Binomial and Monte Carlo vs. BSM exact. Log-scale error analysis.

---

## Project Structure

```
optionlab/
├── src/optionlab/
│   ├── models.py              # Option / Greeks dataclasses
│   ├── pricers/
│   │   ├── base.py            # Abstract Pricer interface
│   │   ├── black_scholes.py   # BSM closed-form + put-call parity check
│   │   ├── binomial.py        # CRR recombining tree
│   │   └── monte_carlo.py     # GBM + antithetic variates
│   ├── implied_vol.py         # Brent root-finder for IV
│   └── vol_smile.py           # Yahoo Finance options chain + smile data
├── tests/
│   ├── test_black_scholes.py
│   ├── test_binomial_mc.py
│   ├── test_implied_vol.py
│   └── test_vol_smile.py
├── examples/
│   └── plot_smile.py
├── app.py                     # Streamlit dashboard
└── pyproject.toml
```

---

## Installation

```bash
git clone https://github.com/YOUR_USERNAME/optionlab.git
cd optionlab
pip install -e .
```

**Run the dashboard:**
```bash
python -m streamlit run app.py
```

**Run tests:**
```bash
pytest
```

---

## Usage

```python
from optionlab import Option, OptionType, BlackScholesPricer

option = Option(
    S=100.0,    # spot price
    K=105.0,    # strike
    T=0.25,     # time to expiry (years)
    r=0.05,     # risk-free rate
    sigma=0.20, # implied volatility
    option_type=OptionType.CALL,
    q=0.0,      # dividend yield
)

pricer = BlackScholesPricer()
price  = pricer.price(option)    # 3.0842...
greeks = pricer.greeks(option)   # Greeks(delta=0.4271, gamma=0.0188, ...)

print(f"Price : ${price:.4f}")
print(f"Delta : {greeks.delta:+.4f}")
print(f"Gamma : {greeks.gamma:.5f}")
print(f"Theta : {greeks.theta:.4f} / day")
print(f"Vega  : {greeks.vega:.4f} / 1% vol")
```

**Implied volatility:**
```python
from optionlab.implied_vol import implied_vol_brent

iv = implied_vol_brent(
    market_price=3.50,
    S=100.0, K=105.0, T=0.25, r=0.05,
    option_type=OptionType.CALL,
)
print(f"IV: {iv:.2%}")
```

---

## Math

### Black-Scholes-Merton

$$C = S e^{-qT} \Phi(d_1) - K e^{-rT} \Phi(d_2)$$

$$P = K e^{-rT} \Phi(-d_2) - S e^{-qT} \Phi(-d_1)$$

$$d_1 = \frac{\ln(S/K) + (r - q + \frac{1}{2}\sigma^2)T}{\sigma\sqrt{T}}, \quad d_2 = d_1 - \sigma\sqrt{T}$$

### Put-Call Parity

$$C - P = S e^{-qT} - K e^{-rT}$$

Verified at every strike in the Chain tab — residual < 10⁻⁶.

### Monte Carlo (GBM)

$$S_T = S_0 \exp\!\left[\left(r - q - \tfrac{1}{2}\sigma^2\right)T + \sigma\sqrt{T}\,Z\right], \quad Z \sim \mathcal{N}(0,1)$$

Antithetic variates: each path Z is paired with −Z, halving variance at no extra cost.

---

## Design Decisions

- **`frozen=True` dataclass for `Option`** — options are value objects; mutability is a bug surface
- **`ABC` for `Pricer`** — callers depend on the interface, not the implementation; models are swappable
- **Brent over Newton as default IV solver** — Newton is faster but diverges on deep OTM options; Brent guarantees convergence when a valid bracket exists
- **Antithetic variates in MC** — halves variance at zero extra simulation cost by exploiting the symmetry of Brownian motion
- **`numba` as optional dependency** — MC inner loop compiles to native code when available; graceful fallback otherwise

---

## Results

Black-Scholes vs Binomial (1000 steps) vs Monte Carlo (100k paths) on ATM call:

| Model | Price | Error vs BS |
|---|---|---|
| Black-Scholes | exact | — |
| Binomial CRR (1000 steps) | converges | < $0.001 |
| Monte Carlo (100k paths) | converges | ~$0.01 (sampling noise) |

Put-call parity residual: `< 1e-6` at every strike.

---

## Tech Stack

**Python 3.11+** · NumPy · SciPy · Pandas · Plotly · Streamlit · yfinance

---

## License

MIT
