"""
app.py  —  optionlab
Run:  python -m streamlit run app.py
"""
from __future__ import annotations
import dataclasses, datetime, math, warnings
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from optionlab import Option, OptionType, BlackScholesPricer, BinomialPricer, MonteCarloPricer
from optionlab.vol_smile import fetch_smile_data, list_expiries
from optionlab.pricers.black_scholes import _d1, _d2, _Phi

st.set_page_config(
    page_title="optionlab",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# Design System  (OLED dark · single accent · Inter · no gradients)
# Source: ui-ux-pro-max  →  Dark Mode OLED + Financial Dashboard
# ─────────────────────────────────────────────────────────────────────────────
BG   = "#020617"   # page background  (slate-950)
S1   = "#0F172A"   # sidebar          (slate-900)
S2   = "#1E293B"   # card/surface     (slate-800)
S3   = "#293548"   # hover/elevated   (slate-700-ish)
BDR  = "rgba(255,255,255,0.06)"
BDR2 = "rgba(255,255,255,0.10)"

T1   = "#F8FAFC"   # primary text     (slate-50)
T2   = "#94A3B8"   # secondary text   (slate-400)
T3   = "#475569"   # muted            (slate-600)

# ── Single accent + semantic colours ─────────────────────────────────────────
GREEN = "#22C55E"   # profit / positive / primary accent  (green-500)
RED   = "#EF4444"   # loss / negative                     (red-500)
BLUE  = "#3B82F6"   # chart primary / info                (blue-500)
AMBER = "#F59E0B"   # breakeven / warning                 (amber-500)
SLATE = "#64748B"   # neutral trace                       (slate-500)

# ── Chart theme ───────────────────────────────────────────────────────────────
PLOT_BG  = S1
GRID_C   = "rgba(255,255,255,0.04)"
PLOTLY   = dict(
    template    = "plotly_dark",
    paper_bgcolor = "rgba(0,0,0,0)",
    plot_bgcolor  = PLOT_BG,
    font        = dict(family="Inter, sans-serif", color=T3, size=11),
    xaxis       = dict(gridcolor=GRID_C, linecolor=BDR, zerolinecolor=GRID_C,
                       tickfont=dict(color=T3, size=10)),
    yaxis       = dict(gridcolor=GRID_C, linecolor=BDR, zerolinecolor=GRID_C,
                       tickfont=dict(color=T3, size=10)),
    margin      = dict(t=44, b=36, l=8, r=8),
    hoverlabel  = dict(bgcolor=S2, bordercolor=BDR2,
                       font=dict(color=T1, family="JetBrains Mono, monospace", size=12)),
    legend      = dict(bgcolor="rgba(0,0,0,0)", bordercolor="rgba(0,0,0,0)",
                       font=dict(color=T2, size=11)),
)

# ── Strategy colour map (for multi-leg P&L lines) ─────────────────────────────
CHART_PAL = [BLUE, AMBER, GREEN, SLATE, RED]

# ─────────────────────────────────────────────────────────────────────────────
# CSS  (injected once)
# ─────────────────────────────────────────────────────────────────────────────
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

*, *::before, *::after {{ box-sizing: border-box; }}

html, body, [class*="css"] {{
    font-family: 'Inter', -apple-system, system-ui, sans-serif !important;
    -webkit-font-smoothing: antialiased;
    -moz-osx-font-smoothing: grayscale;
}}

/* ── Page ── */
.stApp {{
    background-color: {BG};
    color: {T1};
}}
.block-container {{
    padding: 2rem 2.5rem 3rem !important;
    max-width: 1440px !important;
}}

/* ── Sidebar ── */
[data-testid="stSidebar"] {{
    background-color: {S1} !important;
    border-right: 1px solid {BDR} !important;
}}
[data-testid="stSidebar"] .block-container {{
    padding: 1.8rem 1.5rem !important;
}}
[data-testid="stSidebar"] label {{
    color: {T3} !important;
    font-size: 0.73rem !important;
    font-weight: 500 !important;
    letter-spacing: 0.02em !important;
}}
[data-testid="stSidebar"] p {{ color: {T2} !important; font-size: 0.82rem !important; }}

/* Slider — nuke all default colors, uniform dark track */
[data-testid="stSlider"] div[data-baseweb="slider"] * {{
    background: transparent !important;
    box-shadow: none !important;
}}
[data-testid="stSlider"] div[data-baseweb="slider"] > div:first-child {{
    background: rgba(255,255,255,0.08) !important;
    height: 3px !important;
    border-radius: 2px !important;
}}
[data-testid="stSlider"] div[data-baseweb="slider"] > div:first-child > div {{
    background: rgba(255,255,255,0.08) !important;
    height: 3px !important;
}}
/* Thumb */
[data-testid="stSlider"] div[role="slider"] {{
    background: {GREEN} !important;
    border: 2px solid {BG} !important;
    width: 14px !important;
    height: 14px !important;
    border-radius: 50% !important;
    box-shadow: none !important;
}}

/* Radio */
.stRadio label {{ color: {T2} !important; font-size: 0.8rem !important; }}
.stRadio [data-testid="stMarkdownContainer"] p {{ color: {T2} !important; }}

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {{
    background: transparent !important;
    border-bottom: 1px solid {BDR} !important;
    gap: 0 !important;
    padding: 0 !important;
}}
.stTabs [data-baseweb="tab"] {{
    background: transparent !important;
    color: {T3} !important;
    font-size: 0.8rem !important;
    font-weight: 500 !important;
    border: none !important;
    padding: 0.65rem 1.3rem !important;
    border-bottom: 2px solid transparent !important;
    transition: color 0.15s ease !important;
    cursor: pointer !important;
}}
.stTabs [data-baseweb="tab"]:hover {{ color: {T1} !important; }}
.stTabs [aria-selected="true"] {{
    color: {T1} !important;
    border-bottom: 2px solid {GREEN} !important;
    background: transparent !important;
}}

/* ── Buttons ── */
.stButton > button {{
    background: transparent !important;
    color: {T2} !important;
    border: 1px solid {BDR2} !important;
    border-radius: 6px !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 0.79rem !important;
    font-weight: 500 !important;
    padding: 0.38rem 0.9rem !important;
    transition: all 0.15s ease !important;
    cursor: pointer !important;
}}
.stButton > button:hover {{
    background: {S2} !important;
    color: {T1} !important;
    border-color: {BDR2} !important;
}}
.stButton > button[kind="primary"] {{
    background: {GREEN} !important;
    color: #000 !important;
    border: none !important;
    font-weight: 600 !important;
}}
.stButton > button[kind="primary"]:hover {{
    background: #16a34a !important;
}}

/* ── Text input ── */
.stTextInput > div > div > input {{
    background: {S2} !important;
    border: 1px solid {BDR2} !important;
    border-radius: 6px !important;
    color: {T1} !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 0.85rem !important;
    padding: 0.42rem 0.75rem !important;
    transition: border-color 0.15s ease;
}}
.stTextInput > div > div > input:focus {{
    border-color: {GREEN} !important;
    outline: none !important;
    box-shadow: 0 0 0 2px rgba(34,197,94,0.15) !important;
}}
.stTextInput > div > div > input::placeholder {{ color: {T3} !important; }}

/* ── Selectbox ── */
[data-testid="stSelectbox"] > div > div {{
    background: {S2} !important;
    border: 1px solid {BDR2} !important;
    border-radius: 6px !important;
    color: {T1} !important;
}}

/* ── Expander ── */
[data-testid="stExpander"] {{
    background: {S2} !important;
    border: 1px solid {BDR} !important;
    border-radius: 8px !important;
}}
[data-testid="stExpander"] summary {{
    color: {T2} !important;
    font-size: 0.82rem !important;
    font-weight: 500 !important;
}}

/* ── Spinner ── */
.stSpinner > div {{ border-top-color: {GREEN} !important; }}

/* ── Alert ── */
[data-testid="stAlert"] {{
    background: {S2} !important;
    border: 1px solid {BDR} !important;
    border-radius: 8px !important;
    color: {T2} !important;
}}

/* ── Misc ── */
hr {{ border-color: {BDR} !important; margin: 1rem 0 !important; }}
.stMarkdown p {{ color: {T2} !important; line-height: 1.6; font-size: 0.85rem; }}
#MainMenu, footer, header {{ visibility: hidden; }}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# HTML components
# ─────────────────────────────────────────────────────────────────────────────
def metric_card(label: str, value: str, sub: str = "",
                color: str = T2, positive: bool | None = None) -> str:
    """KPI card: flat dark surface, left-accent bar, monospaced value."""
    if positive is True:
        color = GREEN
    elif positive is False:
        color = RED
    rv, gv, bv = int(color[1:3], 16), int(color[3:5], 16), int(color[5:7], 16)
    return f"""
<div style="background:{S2};border:1px solid {BDR};border-left:3px solid {color};
  border-radius:8px;padding:14px 16px;height:100%;">
  <div style="color:{T3};font-size:0.65rem;font-weight:600;letter-spacing:.09em;
    text-transform:uppercase;margin-bottom:8px;">{label}</div>
  <div style="color:{T1};font-size:1.22rem;font-weight:600;
    font-family:'JetBrains Mono',monospace;letter-spacing:-.02em;line-height:1.1;">{value}</div>
  {"" if not sub else f'<div style="color:{T3};font-size:0.67rem;margin-top:5px;">{sub}</div>'}
</div>"""


def tag(text: str, color: str = SLATE) -> str:
    rv, gv, bv = int(color[1:3], 16), int(color[3:5], 16), int(color[5:7], 16)
    return (f'<span style="background:rgba({rv},{gv},{bv},0.15);color:{color};'
            f'border:1px solid rgba({rv},{gv},{bv},0.3);border-radius:4px;'
            f'padding:2px 8px;font-size:0.71rem;font-weight:600;'
            f'letter-spacing:.02em;">{text}</span>')


def section(text: str) -> None:
    st.markdown(
        f'<div style="color:{T3};font-size:0.68rem;font-weight:600;letter-spacing:.09em;'
        f'text-transform:uppercase;margin:1.4rem 0 .75rem;padding-bottom:.5rem;'
        f'border-bottom:1px solid {BDR};">{text}</div>',
        unsafe_allow_html=True,
    )


def info_card(title: str, symbol: str, body: str, color: str = BLUE) -> str:
    return (f'<div style="background:{S2};border:1px solid {BDR};'
            f'border-left:3px solid {color};border-radius:8px;padding:14px 16px;height:100%;">'
            f'<div style="display:flex;align-items:baseline;gap:8px;margin-bottom:8px;">'
            f'<span style="color:{color};font-size:1rem;font-weight:700;'
            f'font-family:JetBrains Mono,monospace;">{symbol}</span>'
            f'<span style="color:{T2};font-size:0.8rem;font-weight:600;">{title}</span></div>'
            f'<div style="color:{T3};font-size:0.75rem;line-height:1.65;">{body}</div></div>')


def html_table(df: pd.DataFrame, accent_col: str | None = None,
               accent_val: str | None = None, accent_color: str = GREEN) -> None:
    """Render DataFrame as styled HTML table — avoids st.dataframe shadow-DOM issues."""
    rv, gv, bv = int(accent_color[1:3], 16), int(accent_color[3:5], 16), int(accent_color[5:7], 16)
    heads = "".join(
        f'<th style="padding:9px 14px;text-align:left;color:{T3};font-size:.67rem;'
        f'font-weight:600;letter-spacing:.08em;text-transform:uppercase;'
        f'border-bottom:1px solid {BDR2};white-space:nowrap;">{c}</th>'
        for c in df.columns
    )
    rows = []
    for _, row in df.iterrows():
        highlight = (accent_col and str(row.get(accent_col, "")) == str(accent_val))
        bg = f"background:rgba({rv},{gv},{bv},0.08);" if highlight else ""
        cells = "".join(
            f'<td style="padding:8px 14px;border-bottom:1px solid {BDR};color:{T1};'
            f'font-family:JetBrains Mono,monospace;font-size:.79rem;'
            f'white-space:nowrap;">{v}</td>'
            for v in row
        )
        rows.append(f'<tr style="{bg}">{cells}</tr>')
    st.markdown(
        f'<div style="border:1px solid {BDR};border-radius:8px;overflow-x:auto;margin:.25rem 0 .5rem;">'
        f'<table style="width:100%;border-collapse:collapse;background:{S2};">'
        f'<thead><tr>{heads}</tr></thead><tbody>{"".join(rows)}</tbody></table></div>',
        unsafe_allow_html=True,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Strategy definitions
# ─────────────────────────────────────────────────────────────────────────────
STRAT_LEGS: dict[str, list[tuple]] = {
    "Bull Call Spread" : [(OptionType.CALL,  0, 1, +1), (OptionType.CALL, +1, 1, -1)],
    "Bear Put Spread"  : [(OptionType.PUT,   0, 1, +1), (OptionType.PUT,  -1, 1, -1)],
    "Long Straddle"    : [(OptionType.CALL,  0, 1, +1), (OptionType.PUT,   0, 1, +1)],
    "Long Strangle"    : [(OptionType.CALL, +1, 1, +1), (OptionType.PUT,  -1, 1, +1)],
    "Iron Condor"      : [(OptionType.PUT,  -1, 1, -1), (OptionType.PUT,  -2, 1, +1),
                          (OptionType.CALL, +1, 1, -1), (OptionType.CALL, +2, 1, +1)],
    "Butterfly"        : [(OptionType.CALL, -1, 1, +1), (OptionType.CALL,  0, 2, -1),
                          (OptionType.CALL, +1, 1, +1)],
}
STRAT_DESC = {
    "Single"          : "One call or put. Profit if spot moves beyond the premium threshold.",
    "Bull Call Spread" : "Buy ATM call, sell OTM call. Capped upside, capped downside. Net debit.",
    "Bear Put Spread"  : "Buy ATM put, sell OTM put. Profits on downward moves. Net debit.",
    "Long Straddle"    : "Buy call + put at same strike. Profits from large moves in either direction.",
    "Long Strangle"    : "Buy OTM call + put. Cheaper than straddle, needs a bigger move.",
    "Iron Condor"      : "Short OTM call spread + short OTM put spread. Earns premium in a range. Net credit.",
    "Butterfly"        : "Buy ITM + OTM calls, sell 2× ATM. Max profit if spot pins the body at expiry.",
}


def build_legs(strategy: str, ot_str: str, K: float, spread: float,
               S: float, T: float, r: float, sigma: float, q: float):
    if strategy == "Single":
        ot  = OptionType.CALL if ot_str == "Call" else OptionType.PUT
        return [(Option(S=S, K=K, T=T, r=r, sigma=sigma, option_type=ot, q=q), 1, +1)]
    legs = []
    for ot, k_off, qty, direction in STRAT_LEGS[strategy]:
        strike = K + k_off * spread
        if strike > 0:
            legs.append((Option(S=S, K=strike, T=T, r=r, sigma=sigma,
                                option_type=ot, q=q), qty, direction))
    return legs


def strategy_payoff(legs, spot: float) -> float:
    total = 0.0
    for opt, qty, d in legs:
        p = max(spot - opt.K, 0) if opt.option_type == OptionType.CALL else max(opt.K - spot, 0)
        total += d * qty * p
    return total


def strategy_value(legs, spot: float, T: float, r: float, sigma: float, q: float) -> float:
    bs = BlackScholesPricer()
    total = 0.0
    for opt, qty, d in legs:
        total += d * qty * bs.price(dataclasses.replace(opt, S=spot, T=T, sigma=sigma))
    return total


# ─────────────────────────────────────────────────────────────────────────────
# Cached computations
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data(ttl=30)
def _greek_curve(S, K, T, r, sigma, q, ot_str):
    spots = np.linspace(max(S * 0.5, 1.0), S * 1.5, 250)
    bs    = BlackScholesPricer()
    ot    = OptionType.CALL if ot_str == "Call" else OptionType.PUT
    base  = Option(S=S, K=K, T=T, r=r, sigma=sigma, option_type=ot, q=q)
    dd, gg, tt, vv, rr, pp = [], [], [], [], [], []
    for s in spots:
        o = dataclasses.replace(base, S=float(s))
        g = bs.greeks(o)
        dd.append(g.delta); gg.append(g.gamma); tt.append(g.theta)
        vv.append(g.vega);  rr.append(g.rho);   pp.append(bs.price(o))
    return spots, dict(Delta=dd, Gamma=gg, Theta=tt, Vega=vv, Rho=rr), pp


@st.cache_data(ttl=30)
def _greek_heatmap(S, K, T, r, q, ot_str, metric: str, n=42):
    sv  = np.linspace(S * 0.65, S * 1.35, n)
    sig = np.linspace(0.04, 0.90, n)
    bs  = BlackScholesPricer()
    ot  = OptionType.CALL if ot_str == "Call" else OptionType.PUT
    Z   = np.zeros((n, n))
    for i, s2 in enumerate(sig):
        for j, s in enumerate(sv):
            opt = Option(S=float(s), K=K, T=T, r=r, sigma=float(s2), option_type=ot, q=q)
            Z[i, j] = bs.price(opt) if metric == "Price" else getattr(bs.greeks(opt), metric.lower())
    return sv, sig, Z


@st.cache_data(ttl=30)
def _option_chain(S, K, T, r, sigma, q, n_each: int = 8):
    bs   = BlackScholesPricer()
    step = max(round(S * 0.025 / 0.5) * 0.5, 1.0)
    strikes = sorted({round(K + i * step, 2) for i in range(-n_each, n_each + 1)})
    rows = []
    for strike in strikes:
        c = Option(S=S, K=strike, T=T, r=r, sigma=sigma, option_type=OptionType.CALL, q=q)
        p = Option(S=S, K=strike, T=T, r=r, sigma=sigma, option_type=OptionType.PUT,  q=q)
        cg, pg = bs.greeks(c), bs.greeks(p)
        rows.append({
            "Strike"  : strike,
            "ATM"     : abs(strike - K) < step * 0.6,
            "C Price" : round(bs.price(c), 3),
            "C Δ"     : round(cg.delta, 3),
            "C Γ"     : round(cg.gamma, 4),
            "C Θ"     : round(cg.theta, 4),
            "C ν"     : round(cg.vega,  4),
            "P Price" : round(bs.price(p), 3),
            "P Δ"     : round(pg.delta, 3),
            "P Γ"     : round(pg.gamma, 4),
            "P Θ"     : round(pg.theta, 4),
            "P ν"     : round(pg.vega,  4),
            "K/S"     : round(strike / S, 3),
        })
    return pd.DataFrame(rows)


@st.cache_data(ttl=30)
def _pnl_matrix(strategy, ot_str, K, spread, S, T_days, r, sigma, q):
    spots     = np.linspace(S * 0.70, S * 1.30, 13)
    day_steps = sorted({T_days, max(1, int(T_days*.75)), max(1, int(T_days*.5)),
                         max(1, int(T_days*.25)), 1, 0}, reverse=True)
    T_   = T_days / 365.0
    legs = build_legs(strategy, ot_str, K, spread, S, T_, r, sigma, q)
    bs   = BlackScholesPricer()
    net  = sum(d * qty * bs.price(opt) for opt, qty, d in legs)
    mat  = np.zeros((len(day_steps), len(spots)))
    for i, d in enumerate(day_steps):
        for j, spot in enumerate(spots):
            val = (strategy_payoff(legs, spot) if d == 0
                   else strategy_value(legs, spot, d / 365.0, r, sigma, q))
            mat[i, j] = val - net
    return mat, [f"${s:.0f}" for s in spots], [f"{d}d" if d else "Expiry" for d in day_steps], net


def _convergence(S, K, T, r, sigma, q, ot_str):
    ot    = OptionType.CALL if ot_str == "Call" else OptionType.PUT
    opt   = Option(S=S, K=K, T=T, r=r, sigma=sigma, option_type=ot, q=q)
    bs_px = BlackScholesPricer().price(opt)
    steps = [1, 2, 5, 10, 20, 50, 100, 250, 500, 1000]
    paths = [100, 500, 1_000, 5_000, 10_000, 50_000, 100_000]
    crr   = [BinomialPricer(steps=n).price(opt) for n in steps]
    mc    = [MonteCarloPricer(n_paths=n, seed=42).price(opt) for n in paths]
    return bs_px, steps, crr, paths, mc


@st.cache_data(ttl=120)
def _smile(ticker, expiry):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return fetch_smile_data(ticker, expiry)


@st.cache_data(ttl=300)
def _expiries(ticker):
    return list_expiries(ticker)


# ─────────────────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(
        f'<div style="margin-bottom:1.8rem;">'
        f'<div style="font-size:1.1rem;font-weight:700;letter-spacing:-.02em;'
        f'color:{T1};margin-bottom:3px;">◈ optionlab</div>'
        f'<div style="color:{T3};font-size:0.67rem;letter-spacing:.1em;'
        f'text-transform:uppercase;">Options Pricing Engine</div></div>',
        unsafe_allow_html=True,
    )

    st.markdown(f'<div style="color:{T3};font-size:.67rem;font-weight:600;letter-spacing:.09em;'
                f'text-transform:uppercase;margin-bottom:.5rem;">Contract</div>',
                unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1: opt_type_str = st.radio("Type",  ["Call", "Put"])
    with c2: model_name   = st.radio("Model", ["Black-Scholes", "Binomial", "Monte Carlo"])

    st.markdown(f'<div style="height:1px;background:{BDR};margin:1rem 0;"></div>',
                unsafe_allow_html=True)
    st.markdown(f'<div style="color:{T3};font-size:.67rem;font-weight:600;letter-spacing:.09em;'
                f'text-transform:uppercase;margin-bottom:.6rem;">Parameters</div>',
                unsafe_allow_html=True)

    S      = st.slider("Spot  S",       10.0, 500.0, 100.0, 0.5,  format="$%.1f")
    K      = st.slider("Strike  K",     10.0, 500.0, 100.0, 0.5,  format="$%.1f")
    T_days = st.slider("Expiry  T",      1,    730,   90,    1,    format="%dd")
    sigma  = st.slider("Volatility σ",  1.0,  150.0, 20.0,  0.5,  format="%.1f%%") / 100
    r      = st.slider("Rate  r",       0.0,   15.0,  5.0,  0.1,  format="%.1f%%") / 100
    q      = st.slider("Div yield  q",  0.0,   10.0,  0.0,  0.1,  format="%.1f%%") / 100

    st.markdown(f'<div style="height:1px;background:{BDR};margin:1rem 0;"></div>',
                unsafe_allow_html=True)

    # Status widget
    moneyness = K / S
    itm  = (opt_type_str == "Call" and S > K) or (opt_type_str == "Put" and S < K)
    atm  = abs(moneyness - 1) < 0.01
    sc   = GREEN if itm else (AMBER if atm else RED)
    st_  = "ITM" if itm else ("ATM" if atm else "OTM")
    intr = max(S - K, 0) if opt_type_str == "Call" else max(K - S, 0)
    try:
        d1_ = _d1(S, K, T_days / 365, r, sigma, q)
        d2_ = _d2(d1_, sigma, T_days / 365)
        p_itm = _Phi(d2_) if opt_type_str == "Call" else _Phi(-d2_)
    except Exception:
        p_itm = float("nan")

    st.markdown(f"""
<div style="background:{S2};border:1px solid {BDR};border-radius:8px;padding:14px;">
  <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:12px;">
    <span style="color:{T3};font-size:.65rem;font-weight:600;
      letter-spacing:.09em;text-transform:uppercase;">Status</span>
    {tag(st_, sc)}
  </div>
  <div style="display:grid;grid-template-columns:1fr 1fr;gap:10px;">
    <div>
      <div style="color:{T3};font-size:.61rem;text-transform:uppercase;
        letter-spacing:.07em;margin-bottom:3px;">Moneyness</div>
      <div style="color:{T1};font-size:.88rem;font-weight:500;
        font-family:'JetBrains Mono',monospace;">{moneyness:.4f}</div>
    </div>
    <div>
      <div style="color:{T3};font-size:.61rem;text-transform:uppercase;
        letter-spacing:.07em;margin-bottom:3px;">Intrinsic</div>
      <div style="color:{T1};font-size:.88rem;font-weight:500;
        font-family:'JetBrains Mono',monospace;">${intr:.2f}</div>
    </div>
    <div>
      <div style="color:{T3};font-size:.61rem;text-transform:uppercase;
        letter-spacing:.07em;margin-bottom:3px;">P(ITM)</div>
      <div style="color:{T1};font-size:.88rem;font-weight:500;
        font-family:'JetBrains Mono',monospace;">{p_itm:.1%}</div>
    </div>
    <div>
      <div style="color:{T3};font-size:.61rem;text-transform:uppercase;
        letter-spacing:.07em;margin-bottom:3px;">Type</div>
      <div style="color:{T1};font-size:.88rem;font-weight:500;
        font-family:'JetBrains Mono',monospace;">{opt_type_str}</div>
    </div>
  </div>
</div>""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Build option
# ─────────────────────────────────────────────────────────────────────────────
T_       = T_days / 365.0
opt_type = OptionType.CALL if opt_type_str == "Call" else OptionType.PUT
option   = Option(S=S, K=K, T=T_, r=r, sigma=sigma, option_type=opt_type, q=q)
pricer   = {"Black-Scholes": BlackScholesPricer(),
            "Binomial"     : BinomialPricer(steps=500),
            "Monte Carlo"  : MonteCarloPricer(n_paths=50_000, seed=42)}[model_name]
price    = pricer.price(option)
greeks   = pricer.greeks(option)
intr_val = max(S - K, 0) if opt_type_str == "Call" else max(K - S, 0)
time_val = price - intr_val


# ─────────────────────────────────────────────────────────────────────────────
# Page header
# ─────────────────────────────────────────────────────────────────────────────
st.markdown(
    f'<div style="display:flex;align-items:baseline;gap:10px;margin-bottom:.25rem;">'
    f'<h1 style="margin:0;font-size:1.5rem;font-weight:700;letter-spacing:-.03em;'
    f'color:{T1};">{opt_type_str} Option</h1>'
    f'&nbsp;{tag(model_name, SLATE)}</div>'
    f'<div style="color:{T3};font-size:.79rem;margin-bottom:1.5rem;">'
    f'S&nbsp;<span style="color:{T2};font-family:JetBrains Mono,monospace">${S:.2f}</span>'
    f'&ensp;K&nbsp;<span style="color:{T2};font-family:JetBrains Mono,monospace">${K:.2f}</span>'
    f'&ensp;T&nbsp;<span style="color:{T2};font-family:JetBrains Mono,monospace">{T_days}d</span>'
    f'&ensp;σ&nbsp;<span style="color:{T2};font-family:JetBrains Mono,monospace">{sigma*100:.1f}%</span>'
    f'&ensp;r&nbsp;<span style="color:{T2};font-family:JetBrains Mono,monospace">{r*100:.1f}%</span>'
    f'&ensp;P(ITM)&nbsp;<span style="color:{T2};font-family:JetBrains Mono,monospace">{p_itm:.1%}</span>'
    f'</div>',
    unsafe_allow_html=True,
)

# Metric strip
c1, c2, c3, c4, c5, c6 = st.columns(6)
for col, lbl, val, sub, clr in [
    (c1, "Fair Value",  f"${price:.4f}",         model_name,          GREEN),
    (c2, "Delta  Δ",    f"{greeks.delta:+.4f}",   "dV / dS",           BLUE),
    (c3, "Gamma  Γ",    f"{greeks.gamma:.5f}",    "d²V / dS²",         BLUE),
    (c4, "Theta  Θ",    f"{greeks.theta:.4f}",    "per day",           AMBER if greeks.theta > 0 else RED),
    (c5, "Vega  ν",     f"{greeks.vega:.4f}",     "per 1% vol",        SLATE),
    (c6, "Rho  ρ",      f"{greeks.rho:.4f}",      "per 1% rate",       SLATE),
]:
    with col:
        st.markdown(metric_card(lbl, val, sub, clr), unsafe_allow_html=True)

st.markdown("<div style='height:1.5rem'></div>", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Tabs
# ─────────────────────────────────────────────────────────────────────────────
TAB1, TAB2, TAB3, TAB4, TAB5, TAB6 = st.tabs(
    ["Payoff", "Greeks", "Heatmap", "Chain", "Vol Smile", "Models"]
)
H = 420


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1  ·  Payoff + Strategy Builder
# ══════════════════════════════════════════════════════════════════════════════
with TAB1:
    section("Strategy")
    ALL_STRATS = ["Single"] + list(STRAT_LEGS.keys())
    if "strategy" not in st.session_state:
        st.session_state["strategy"] = "Single"

    btn_cols = st.columns(len(ALL_STRATS))
    for col, name in zip(btn_cols, ALL_STRATS):
        with col:
            if st.button(name, key=f"s_{name}", use_container_width=True):
                st.session_state["strategy"] = name
                st.rerun()

    strategy = st.session_state["strategy"]
    st.markdown(f'<p style="color:{T3};font-size:.76rem;margin:.3rem 0 .8rem;">'
                f'{STRAT_DESC[strategy]}</p>', unsafe_allow_html=True)

    spread = (st.slider("Spread width ($)", 1.0, max(S * 0.25, 5.0),
                        max(S * 0.05, 5.0), 0.5, format="$%.1f", key="spread")
              if strategy != "Single" else 5.0)

    legs     = build_legs(strategy, opt_type_str, K, spread, S, T_, r, sigma, q)
    bs_      = BlackScholesPricer()
    net_prem = sum(d * qty * bs_.price(opt) for opt, qty, d in legs)

    # P&L chart
    section("P&L Diagram")
    spots_pl  = np.linspace(max(S * 0.5, 1.0), S * 1.55, 400)
    exp_pnl   = np.array([strategy_payoff(legs, float(s)) - net_prem for s in spots_pl])
    cur_pnl   = np.array([strategy_value(legs, float(s), T_, r, sigma, q) - net_prem
                           for s in spots_pl])

    fig = go.Figure()
    # Profit / loss fill zones
    fig.add_trace(go.Scatter(
        x=spots_pl, y=np.where(exp_pnl >= 0, exp_pnl, 0),
        fill="tozeroy", fillcolor="rgba(34,197,94,0.07)",
        line=dict(width=0), showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(
        x=spots_pl, y=np.where(exp_pnl < 0, exp_pnl, 0),
        fill="tozeroy", fillcolor="rgba(239,68,68,0.07)",
        line=dict(width=0), showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(
        x=spots_pl, y=exp_pnl, name="P&L at Expiry",
        line=dict(color=T1, width=1.8),
        hovertemplate="Spot $%{x:.2f}  |  P&L $%{y:.4f}<extra></extra>"))
    fig.add_trace(go.Scatter(
        x=spots_pl, y=cur_pnl, name="Current Value",
        line=dict(color=SLATE, width=1.4, dash="dot"),
        hovertemplate="Spot $%{x:.2f}  |  $%{y:.4f}<extra></extra>"))
    drawn: set = set()
    for opt_, qty_, dir_ in legs:
        if opt_.K not in drawn:
            drawn.add(opt_.K)
            fig.add_vline(x=opt_.K, line_dash="dot", line_color=T3, line_width=1,
                          annotation_text=f"{opt_.K:.0f}",
                          annotation_font_color=T3, annotation_font_size=9)
    fig.add_vline(x=S, line_color=T2, line_width=1, line_dash="dash",
                  annotation_text=f"S={S:.0f}", annotation_font_color=T2, annotation_font_size=9)
    fig.add_hline(y=0, line_color=BDR2, line_width=1)
    fig.update_layout(
        **PLOTLY, height=H,
        title=dict(text=f"{strategy}  ·  net premium "
                        f"<span style='color:{'#22C55E' if net_prem>0 else '#EF4444'}'>"
                        f"${net_prem:+.3f}</span>",
                   font=dict(size=12, color=T2)),
        xaxis_title="Spot at Expiry ($)", yaxis_title="P&L ($)",
        hovermode="x unified",
    )
    st.plotly_chart(fig, use_container_width=True)

    # Stat cards
    max_p   = float(np.max(exp_pnl))
    max_l   = float(np.min(exp_pnl))
    be_pts  = []
    for i in range(len(exp_pnl) - 1):
        if exp_pnl[i] * exp_pnl[i + 1] <= 0 and exp_pnl[i] != exp_pnl[i + 1]:
            x0, x1, y0, y1 = spots_pl[i], spots_pl[i+1], exp_pnl[i], exp_pnl[i+1]
            be_pts.append(x0 - y0 * (x1 - x0) / (y1 - y0))
    be_str = "  /  ".join(f"${b:.2f}" for b in be_pts) if be_pts else "—"

    pa, pb, pc, pd_ = st.columns(4)
    for col, lbl, val, pos in [
        (pa,  "Net Premium",  f"${net_prem:+.4f}",
              None if abs(net_prem) < .001 else (net_prem > 0)),
        (pb,  "Breakeven(s)", be_str,  None),
        (pc,  "Max Profit",   "Unlimited" if max_p > 999 else f"${max_p:.2f}", True),
        (pd_, "Max Loss",     "Unlimited" if max_l < -999 else f"${max_l:.2f}", False),
    ]:
        with col:
            st.markdown(metric_card(lbl, val, positive=pos), unsafe_allow_html=True)

    # P&L Scenario Matrix
    st.markdown("<div style='height:.75rem'></div>", unsafe_allow_html=True)
    section("P&L Scenario Matrix  —  spot × time remaining")
    st.markdown(f'<p style="color:{T3};font-size:.76rem;margin-bottom:.6rem;">'
                f'Each cell is the net P&L of the strategy at that (spot, days remaining) point. '
                f'Green = profit, red = loss. Bottom row = P&L at expiry.</p>',
                unsafe_allow_html=True)
    try:
        mat, col_lbs, row_lbs, _ = _pnl_matrix(
            strategy, opt_type_str, K, spread, S, T_days, r, sigma, q)
        amax = max(abs(mat.min()), abs(mat.max()), 0.01)
        fig_m = go.Figure(data=go.Heatmap(
            z=mat, x=col_lbs, y=row_lbs,
            zmin=-amax, zmax=amax,
            colorscale=[[0, RED], [0.5, S2], [1, GREEN]],
            colorbar=dict(
                title=dict(text="P&L ($)", font=dict(color=T2, size=11)),
                tickfont=dict(color=T2, size=10),
                outlinecolor=BDR, outlinewidth=1, bgcolor="rgba(0,0,0,0)"),
            text=[[f"${v:.2f}" for v in row] for row in mat],
            texttemplate="%{text}",
            textfont=dict(size=9, family="JetBrains Mono"),
            hovertemplate="Spot %{x}  |  %{y}<br>P&L <b>$%{z:.4f}</b><extra></extra>",
        ))
        fig_m.update_layout(**PLOTLY, height=260,
                            title=dict(text="P&L Matrix",
                                       font=dict(size=12, color=T2)),
                            xaxis_title="Spot", yaxis_title="Time Remaining")
        st.plotly_chart(fig_m, use_container_width=True)
    except Exception as e:
        st.info(f"Matrix unavailable: {e}")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2  ·  Greeks
# ══════════════════════════════════════════════════════════════════════════════
with TAB2:
    spots_g, gcurves, pcurve = _greek_curve(S, K, T_, r, sigma, q, opt_type_str)
    # Minimal single-hue palette: all charts share the same blue, differentiated by opacity
    GCLR = dict(Delta=BLUE, Gamma=BLUE, Theta=RED, Vega=GREEN, Rho=SLATE)

    mode = st.radio("", ["All Greeks", "Single Greek"], horizontal=True, key="gv")

    if mode == "All Greeks":
        fig = make_subplots(rows=2, cols=3,
                            subplot_titles=list(gcurves.keys()) + ["Price"],
                            vertical_spacing=0.18, horizontal_spacing=0.08)
        for (row, col), (name, vals) in zip([(1,1),(1,2),(1,3),(2,1),(2,2)], gcurves.items()):
            clr = GCLR[name]
            rv, gv, bv = int(clr[1:3],16), int(clr[3:5],16), int(clr[5:7],16)
            fig.add_trace(go.Scatter(x=spots_g, y=vals, name=name,
                                     line=dict(color=clr, width=1.8),
                                     fill="tozeroy",
                                     fillcolor=f"rgba({rv},{gv},{bv},0.05)"),
                          row=row, col=col)
            fig.add_vline(x=S, line_dash="dot", line_color=T3, line_width=1, row=row, col=col)
        fig.add_trace(go.Scatter(x=spots_g, y=pcurve, name="Price",
                                  line=dict(color=T2, width=1.8),
                                  fill="tozeroy", fillcolor="rgba(100,116,139,0.05)"),
                      row=2, col=3)
        fig.add_vline(x=S, line_dash="dot", line_color=T3, line_width=1, row=2, col=3)
        base = {k: v for k, v in PLOTLY.items() if k not in ("xaxis","yaxis")}
        fig.update_layout(**base, height=500, showlegend=False,
                          title=dict(text="Greeks & Price vs Spot  (Black-Scholes analytical)",
                                     font=dict(size=13, color=T2)))
        for i in range(1, 7):
            s = "" if i == 1 else str(i)
            fig.update_layout(**{
                f"xaxis{s}": dict(gridcolor=GRID_C, linecolor=BDR, tickfont=dict(color=T3, size=10)),
                f"yaxis{s}": dict(gridcolor=GRID_C, linecolor=BDR, tickfont=dict(color=T3, size=10)),
            })
    else:
        chosen = st.selectbox("Greek", list(gcurves.keys()), key="gs")
        clr = GCLR[chosen]
        rv, gv, bv = int(clr[1:3],16), int(clr[3:5],16), int(clr[5:7],16)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=spots_g, y=gcurves[chosen], name=chosen,
                                  line=dict(color=clr, width=2),
                                  fill="tozeroy", fillcolor=f"rgba({rv},{gv},{bv},0.07)"))
        fig.add_vline(x=S, line_dash="dot", line_color=T3, line_width=1,
                      annotation_text=f"S={S:.0f}", annotation_font_color=T3, annotation_font_size=10)
        fig.update_layout(**PLOTLY, height=H,
                          title=dict(text=f"{chosen} vs Spot", font=dict(size=13, color=T2)),
                          xaxis_title="Spot ($)", yaxis_title=chosen)
    st.plotly_chart(fig, use_container_width=True)

    # Current values table
    section("Current values")
    html_table(pd.DataFrame({
        "Greek":       ["Delta Δ", "Gamma Γ", "Theta Θ", "Vega ν", "Rho ρ"],
        "Value":       [f"{greeks.delta:+.6f}", f"{greeks.gamma:.6f}", f"{greeks.theta:.6f}",
                        f"{greeks.vega:.6f}", f"{greeks.rho:.6f}"],
        "Interpretation": [
            f"${greeks.delta:+.4f} per $1 spot move",
            f"Delta shifts {greeks.gamma:.4f} per $1 move",
            f"-${abs(greeks.theta):.4f} per calendar day",
            f"${greeks.vega:.4f} per 1% implied vol move",
            f"${greeks.rho:.4f} per 1% rate move",
        ],
    }))

    # Greek explanation cards
    section("What the Greeks mean")
    EXPLAIN = [
        ("Delta", "Δ", BLUE,
         "How much the option price changes for a <b>$1 move in the underlying</b>. "
         "Delta of 0.60 → gains $0.60 per $1 rise. Think of it as the "
         "option's stock-equivalent position size."),
        ("Gamma", "Γ", BLUE,
         "How fast delta changes as the stock moves. Peaks sharply at ATM and near expiry. "
         "Long options = long gamma (you gain convexity). "
         "Short options = short gamma (you're hurt by large moves)."),
        ("Theta", "Θ", RED,
         "Daily time decay — value lost just from one day passing. "
         "Almost always negative for buyers. A theta of −0.05 means you lose $0.05/day. "
         "Time is the enemy of buyers, the ally of sellers."),
        ("Vega", "ν", GREEN,
         "Sensitivity to <b>implied volatility</b>. Vega of 0.15 → gains $0.15 per 1% IV rise. "
         "Long options are long vega. Highest for ATM options with more time remaining."),
        ("Rho", "ρ", SLATE,
         "Sensitivity to the <b>risk-free rate</b>. Calls have positive rho; puts have negative. "
         "Less important for short-dated options — matters most for long-dated LEAPS."),
    ]
    cols_ex = st.columns(3)
    for i, (name, sym, clr, body) in enumerate(EXPLAIN):
        with cols_ex[i % 3]:
            st.markdown(info_card(name, sym, body, clr), unsafe_allow_html=True)
            st.markdown("<div style='height:.5rem'></div>", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3  ·  Heatmap
# ══════════════════════════════════════════════════════════════════════════════
with TAB3:
    METRIC_DESC = {
        "Price" : "Option price across the spot-volatility space.",
        "Delta" : "Directional exposure. Approaches 1 (deep ITM) or 0 (deep OTM) for calls.",
        "Gamma" : "Convexity — peaks at ATM. High gamma = delta changes fast near this zone.",
        "Theta" : "Time decay landscape. Deep OTM and deep ITM decay slower than ATM.",
        "Vega"  : "Vol sensitivity — peaks at ATM and fades toward the wings.",
    }
    col_sel, col_desc = st.columns([2, 5])
    with col_sel:
        metric = st.selectbox("Surface metric",
                              ["Price", "Delta", "Gamma", "Theta", "Vega"], key="hm")
    with col_desc:
        st.markdown(f'<p style="color:{T3};font-size:.78rem;margin-top:.35rem;">'
                    f'{METRIC_DESC[metric]}</p>', unsafe_allow_html=True)

    sv, sig_v, Z = _greek_heatmap(S, K, T_, r, q, opt_type_str, metric)

    # Colorscale: 2-tone per metric, no rainbow
    CSCALES = {
        "Price" : [[0, S1], [1, GREEN]],
        "Delta" : [[0, RED],  [0.5, S2], [1, GREEN]],
        "Gamma" : [[0, S1], [1, BLUE]],
        "Theta" : [[0, RED],  [1, S1]],
        "Vega"  : [[0, S1], [1, BLUE]],
    }
    fig = go.Figure(data=go.Heatmap(
        z=Z, x=np.round(sv, 1), y=np.round(sig_v * 100, 1),
        colorscale=CSCALES[metric],
        colorbar=dict(title=dict(text=metric, font=dict(color=T2, size=11)),
                      tickfont=dict(color=T2, size=10),
                      outlinecolor=BDR, outlinewidth=1, bgcolor="rgba(0,0,0,0)"),
        hovertemplate=f"Spot $%{{x:.1f}}  ·  Vol %{{y:.0f}}%<br>{metric} <b>%{{z:.5f}}</b><extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=[S], y=[sigma * 100], mode="markers+text",
        marker=dict(color=T1, size=12, symbol="x-thin", line=dict(width=2.5, color=T1)),
        text=["   you"], textfont=dict(color=T1, size=10),
        textposition="middle right", showlegend=False,
    ))
    fig.update_layout(**PLOTLY, height=490,
                      title=dict(text=f"{opt_type_str}  ·  {metric}  —  Spot × Volatility",
                                 font=dict(size=13, color=T2)),
                      xaxis_title="Spot ($)", yaxis_title="Volatility (%)")
    st.plotly_chart(fig, use_container_width=True)

    # Horizontal slice at current vol
    section(f"{metric} vs Spot  (σ = {sigma*100:.0f}%)")
    idx   = int(np.argmin(np.abs(sig_v - sigma)))
    slice_y = Z[idx, :]
    clr_s = GCLR.get(metric, BLUE) if metric != "Price" else GREEN
    rv_, gv_, bv_ = int(clr_s[1:3],16), int(clr_s[3:5],16), int(clr_s[5:7],16)
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(
        x=np.round(sv, 1), y=slice_y, name=metric,
        line=dict(color=clr_s, width=1.8),
        fill="tozeroy", fillcolor=f"rgba({rv_},{gv_},{bv_},0.06)",
        hovertemplate=f"Spot $%{{x:.1f}}<br>{metric} %{{y:.5f}}<extra></extra>"))
    fig2.add_vline(x=S, line_dash="dot", line_color=T3, line_width=1,
                   annotation_text=f"S={S:.0f}", annotation_font_color=T3)
    _slim = {k: v for k, v in PLOTLY.items() if k != "margin"}
    fig2.update_layout(**_slim, height=240,
                       xaxis_title="Spot ($)", yaxis_title=metric,
                       margin=dict(t=20, b=36, l=8, r=8))
    st.plotly_chart(fig2, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 4  ·  Option Chain
# ══════════════════════════════════════════════════════════════════════════════
with TAB4:
    st.markdown(f'<p style="color:{T2};font-size:.83rem;margin-bottom:.8rem;">'
                f'Theoretical chain computed with Black-Scholes at your current parameters. '
                f'ATM row is highlighted. Helps visualise how price and Greeks shift across strikes.</p>',
                unsafe_allow_html=True)

    n_strikes = st.slider("Strikes each side", 3, 12, 7, 1, key="chain_n")
    chain_df  = _option_chain(S, K, T_, r, sigma, q, n_each=n_strikes)

    section("Calls")
    call_df = chain_df[["ATM", "Strike", "K/S", "C Price", "C Δ", "C Γ", "C Θ", "C ν"]].copy()
    call_df["ATM"] = call_df["ATM"].map({True: "◈", False: ""})
    call_df.columns = ["", "Strike", "K/S", "Price", "Δ Delta", "Γ Gamma", "Θ Theta", "ν Vega"]
    html_table(call_df, accent_col="", accent_val="◈", accent_color=GREEN)

    section("Puts")
    put_df = chain_df[["ATM", "Strike", "K/S", "P Price", "P Δ", "P Γ", "P Θ", "P ν"]].copy()
    put_df["ATM"] = put_df["ATM"].map({True: "◈", False: ""})
    put_df.columns = ["", "Strike", "K/S", "Price", "Δ Delta", "Γ Gamma", "Θ Theta", "ν Vega"]
    html_table(put_df, accent_col="", accent_val="◈", accent_color=GREEN)

    # Delta chart
    section("Delta vs Strike")
    fig_ch = go.Figure()
    fig_ch.add_trace(go.Scatter(
        x=chain_df["Strike"], y=chain_df["C Δ"], name="Call Δ",
        line=dict(color=GREEN, width=1.8), mode="lines+markers",
        marker=dict(size=5),
        hovertemplate="K=$%{x:.2f}  |  Δ=%{y:.4f}<extra>Call</extra>"))
    fig_ch.add_trace(go.Scatter(
        x=chain_df["Strike"], y=chain_df["P Δ"], name="Put Δ",
        line=dict(color=RED, width=1.8), mode="lines+markers",
        marker=dict(size=5),
        hovertemplate="K=$%{x:.2f}  |  Δ=%{y:.4f}<extra>Put</extra>"))
    for yv in [0.5, -0.5]:
        fig_ch.add_hline(y=yv, line_dash="dot", line_color=T3, line_width=1)
    fig_ch.add_vline(x=S, line_dash="dash", line_color=T2, line_width=1,
                     annotation_text=f"S={S:.0f}", annotation_font_color=T2)
    fig_ch.update_layout(**PLOTLY, height=300,
                          title=dict(text="Delta across strikes  (±0.5 = ATM)",
                                     font=dict(size=12, color=T2)),
                          xaxis_title="Strike ($)", yaxis_title="Delta")
    st.plotly_chart(fig_ch, use_container_width=True)

    # Put-call parity
    section("Put-Call Parity Check  —  C − P = S·e⁻ᵠᵀ − K·e⁻ʳᵀ")
    parity = []
    for _, row in chain_df.iterrows():
        theory = S * math.exp(-q * T_) - row["Strike"] * math.exp(-r * T_)
        resid  = (row["C Price"] - row["P Price"]) - theory
        parity.append({
            "Strike":             row["Strike"],
            "C − P":              round(row["C Price"] - row["P Price"], 5),
            "S·e⁻ᵠᵀ − K·e⁻ʳᵀ": round(theory, 5),
            "Residual":           round(resid, 8),
            "Pass":               "✓" if abs(resid) < 1e-6 else "✗",
        })
    html_table(pd.DataFrame(parity), accent_col="Pass", accent_val="✓", accent_color=GREEN)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 5  ·  Vol Smile
# ══════════════════════════════════════════════════════════════════════════════
with TAB5:
    st.markdown(f'<p style="color:{T2};font-size:.83rem;margin-bottom:1rem;">'
                f'Live options chain → implied volatility solved at every strike using '
                f'our Brent root-finder. Also shows the ATM vol term structure across expiries.</p>',
                unsafe_allow_html=True)

    ci, cb = st.columns([3, 1])
    with ci:
        ticker = st.text_input("Ticker", value="SPY",
                               placeholder="SPY · AAPL · NVDA · TSLA",
                               label_visibility="collapsed").upper().strip()
    with cb:
        fetch_btn = st.button("Fetch", type="primary", use_container_width=True)

    if fetch_btn or st.session_state.get("_tk") == ticker:
        st.session_state["_tk"] = ticker
        with st.spinner(f"Loading {ticker}…"):
            try:
                all_exp  = _expiries(ticker)
                cutoff   = datetime.date.today() + datetime.timedelta(days=6)
                expiries = [e for e in all_exp
                            if datetime.date.fromisoformat(e) > cutoff][:6]
                if not expiries:
                    st.error("No upcoming expiries."); st.stop()

                # Use a simple sequential palette — no rainbow
                SMILE_PAL = [BLUE, GREEN, AMBER, SLATE, RED]
                fc, fp    = go.Figure(), go.Figure()
                spot_ref  = rate_ref = None
                rows      = []
                term_rows = []

                for i, exp in enumerate(expiries[:4]):
                    try:
                        calls, puts, spot, Texp, rate = _smile(ticker, exp)
                        spot_ref, rate_ref = spot, rate
                    except Exception as e:
                        st.warning(f"Skipped {exp}: {e}"); continue

                    days  = round(Texp * 365)
                    label = f"{exp}  ({days}d)"
                    clr   = SMILE_PAL[i % len(SMILE_PAL)]

                    for fsm, df in [(fc, calls), (fp, puts)]:
                        if not df.empty:
                            fsm.add_trace(go.Scatter(
                                x=df["strike"] / spot, y=df["impl_vol"] * 100,
                                name=label, line=dict(color=clr, width=1.6),
                                mode="lines+markers", marker=dict(size=4, color=clr),
                                hovertemplate=f"K/S=%{{x:.3f}}  |  IV=%{{y:.1f}}%<extra>{label}</extra>"))

                    for df, side in [(calls, "Call"), (puts, "Put")]:
                        if not df.empty:
                            atm_i  = (df["strike"] - spot).abs().idxmin()
                            atm_iv = df.loc[atm_i, "impl_vol"] * 100
                            rows.append({
                                "Expiry": exp, "Days": days, "Side": side,
                                "ATM IV": f"{atm_iv:.1f}%",
                                "Min IV": f"{df['impl_vol'].min()*100:.1f}%",
                                "Max IV": f"{df['impl_vol'].max()*100:.1f}%",
                                "Strikes": len(df),
                            })
                            if side == "Call":
                                term_rows.append({"Days": days, "ATM IV": atm_iv,
                                                   "Expiry": exp, "color": clr})

                for fsm, side in [(fc, "Calls"), (fp, "Puts")]:
                    fsm.add_vline(x=1.0, line_dash="dot", line_color=T3, line_width=1,
                                  annotation_text="ATM", annotation_font_color=T3,
                                  annotation_font_size=9)
                    fsm.update_layout(**PLOTLY, height=360,
                                       title=dict(text=f"{ticker} — {side}  ·  IV Smile",
                                                  font=dict(size=13, color=T2)),
                                       xaxis_title="Moneyness (K / S)",
                                       yaxis_title="Implied Vol (%)",
                                       hovermode="x unified")

                col_c, col_p = st.columns(2)
                with col_c: st.plotly_chart(fc, use_container_width=True)
                with col_p: st.plotly_chart(fp, use_container_width=True)

                if len(term_rows) >= 2:
                    section("Volatility Term Structure  —  ATM IV vs days to expiry")
                    st.markdown(f'<p style="color:{T3};font-size:.76rem;margin-bottom:.5rem;">'
                                f'Upward slope = vol expected to rise (normal contango). '
                                f'Downward slope = near-term stress priced in (inversion).</p>',
                                unsafe_allow_html=True)
                    tsorted = sorted(term_rows, key=lambda x: x["Days"])
                    fig_ts  = go.Figure()
                    fig_ts.add_trace(go.Scatter(
                        x=[r["Days"] for r in tsorted],
                        y=[r["ATM IV"] for r in tsorted],
                        line=dict(color=BLUE, width=1.8), mode="lines+markers",
                        marker=dict(size=8, color=[r["color"] for r in tsorted],
                                    line=dict(width=1.5, color=T1)),
                        text=[r["Expiry"] for r in tsorted],
                        hovertemplate="%{text}  |  %{x}d  |  IV %{y:.2f}%<extra></extra>"))
                    fig_ts.update_layout(**PLOTLY, height=280,
                                          title=dict(text="ATM IV vs Days to Expiry",
                                                     font=dict(size=12, color=T2)),
                                          xaxis_title="Days to Expiry",
                                          yaxis_title="ATM Implied Vol (%)")
                    st.plotly_chart(fig_ts, use_container_width=True)

                if spot_ref:
                    st.markdown(
                        f'<p style="color:{T3};font-size:.73rem;">'
                        f'Spot <span style="color:{T2}">${spot_ref:.2f}</span>'
                        f'&emsp;·&emsp;Risk-free <span style="color:{T2}">{rate_ref:.2%}</span>'
                        f'&emsp;·&emsp;Source: Yahoo Finance</p>',
                        unsafe_allow_html=True)
                if rows:
                    section("Summary")
                    html_table(pd.DataFrame(rows))

            except Exception as e:
                st.error(f"Could not load {ticker}: {e}")
    else:
        st.markdown(
            f'<div style="background:{S2};border:1px solid {BDR};border-radius:8px;'
            f'padding:2.8rem 2rem;text-align:center;margin-top:.5rem;">'
            f'<div style="color:{T3};font-size:1.8rem;margin-bottom:1rem;">—</div>'
            f'<div style="color:{T1};font-size:.95rem;font-weight:600;margin-bottom:.5rem;">'
            f'Enter a ticker and click Fetch</div>'
            f'<div style="color:{T3};font-size:.8rem;line-height:1.7;">'
            f'Live options chain&emsp;·&emsp;Brent IV solver&emsp;·&emsp;Vol smile + term structure'
            f'</div></div>',
            unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 6  ·  Models
# ══════════════════════════════════════════════════════════════════════════════
with TAB6:
    st.markdown(
        f'<p style="color:{T2};font-size:.83rem;margin-bottom:.8rem;">'
        f'Black-Scholes is the closed-form exact solution. '
        f'Binomial (CRR tree) and Monte Carlo (GBM + antithetic variates) '
        f'are numerical — they converge to the same price as resolution increases.</p>',
        unsafe_allow_html=True)

    theory_cols = st.columns(3)
    for col, lbl, sym, clr, body in [
        (theory_cols[0], "Black-Scholes", "BSM", GREEN,
         "Closed-form solution (Black, Scholes, Merton 1973). Assumes log-normal returns, "
         "constant vol, no dividends. Exact under these assumptions. O(1) complexity."),
        (theory_cols[1], "Binomial Tree", "CRR", BLUE,
         "Cox-Ross-Rubinstein (1979) recombining lattice. Backward induction over up/down nodes. "
         "Naturally handles American exercise. Converges to BS as steps → ∞. O(n²)."),
        (theory_cols[2], "Monte Carlo", "MC", SLATE,
         "Simulates GBM price paths, averages discounted payoffs. Uses antithetic variates "
         "to halve variance. Flexible for exotic payoffs. Converges as O(1/√n)."),
    ]:
        with col:
            st.markdown(info_card(lbl, sym, body, clr), unsafe_allow_html=True)

    st.markdown("<div style='height:.8rem'></div>", unsafe_allow_html=True)
    clr_col, _ = st.columns([1, 7])
    with clr_col:
        if st.button("Clear cache"):
            st.cache_data.clear(); st.rerun()

    with st.spinner("Computing convergence…"):
        bs_px, steps, crr_px, paths, mc_px = _convergence(S, K, T_, r, sigma, q, opt_type_str)

    crr_err = [abs(p - bs_px) for p in crr_px]
    mc_err  = [abs(p - bs_px) for p in mc_px]

    def _conv(xs, ys, bs, clr, xt, title):
        f = go.Figure()
        f.add_trace(go.Scatter(x=xs, y=ys, name="Model",
                               line=dict(color=clr, width=1.8), mode="lines+markers",
                               marker=dict(size=5, color=clr),
                               hovertemplate=f"%{{x}} → $%{{y:.5f}}<extra></extra>"))
        f.add_trace(go.Scatter(x=xs, y=[bs] * len(xs), name="BS exact",
                               line=dict(color=T3, width=1.2, dash="dash"),
                               hovertemplate=f"BS ${bs:.5f}<extra></extra>"))
        f.update_layout(**PLOTLY, height=290,
                        title=dict(text=title, font=dict(size=12, color=T2)),
                        xaxis_title=xt, yaxis_title="Price ($)")
        return f

    def _err(xs, errs, clr, xt, title, note):
        rv, gv, bv = int(clr[1:3],16), int(clr[3:5],16), int(clr[5:7],16)
        f = go.Figure()
        f.add_trace(go.Scatter(x=xs, y=[max(e, 1e-7) for e in errs], name="|Error|",
                               line=dict(color=clr, width=1.8), mode="lines+markers",
                               marker=dict(size=5, color=clr),
                               fill="tozeroy", fillcolor=f"rgba({rv},{gv},{bv},0.06)",
                               hovertemplate=f"%{{x}} → $%{{y:.6f}}<extra></extra>"))
        f.update_layout(**PLOTLY, height=260,
                        title=dict(text=title, font=dict(size=12, color=T2)),
                        xaxis_title=xt, yaxis_title="|Error| ($)", yaxis_type="log",
                        annotations=[dict(text=note, xref="paper", yref="paper",
                                          x=0.99, y=0.98, showarrow=False,
                                          font=dict(color=T3, size=10),
                                          xanchor="right", yanchor="top")])
        return f

    section("Convergence to Black-Scholes")
    ca, cb_ = st.columns(2)
    with ca:  st.plotly_chart(_conv(steps, crr_px, bs_px, BLUE,  "Tree Steps", "Binomial — Convergence"), use_container_width=True)
    with cb_: st.plotly_chart(_conv(paths, mc_px,  bs_px, GREEN, "Paths",      "Monte Carlo — Convergence"), use_container_width=True)

    section("Absolute error  (log scale)")
    ea, eb_ = st.columns(2)
    with ea:  st.plotly_chart(_err(steps, crr_err, BLUE,  "Steps", "Binomial |Error|", "Zigzag = alternating over/under"), use_container_width=True)
    with eb_: st.plotly_chart(_err(paths, mc_err,  GREEN, "Paths", "Monte Carlo |Error|", "Noise = sampling variance"), use_container_width=True)

    section("Summary")
    html_table(pd.DataFrame({
        "Model":      ["Black-Scholes", "Binomial (1000 steps)", "Monte Carlo (100k paths)"],
        "Price":      [f"${bs_px:.6f}", f"${crr_px[-1]:.6f}", f"${mc_px[-1]:.6f}"],
        "|Error|":    ["exact (0)", f"${crr_err[-1]:.6f}", f"${mc_err[-1]:.6f}"],
        "Method":     ["Closed-form formula",
                       "Backward induction, CRR recombining tree",
                       "GBM + antithetic variates"],
        "Complexity": ["O(1)", "O(n²)", "O(n · m)"],
    }), accent_col="Model", accent_val="Black-Scholes", accent_color=GREEN)

    with st.expander("Raw numbers"):
        r1, r2 = st.columns(2)
        with r1:
            st.write(f"**BS: ${bs_px:.6f}**")
            for n, p in zip(steps, crr_px):
                st.write(f"`{n:5d}` steps  `${p:.6f}`  err `${abs(p-bs_px):.6f}`")
        with r2:
            for n, p in zip(paths, mc_px):
                st.write(f"`{n:7,}` paths  `${p:.6f}`  err `${abs(p-bs_px):.6f}`")
