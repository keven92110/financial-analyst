"""Option pricing, payoff calculation, and strategy templates.

Workflow
--------
1. Black-Scholes pricing for synthetic option chain (ATM ± offsets).
2. Payoff at expiry for each leg type (long/short, call/put).
3. Strategy templates: single legs, vertical spreads, straddles, strangles,
   iron condor, iron butterfly.
4. Evaluator: given (legs, empirical distribution of S_T at expiry) →
   expected P&L, probability of profit, max gain/loss, breakeven points.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional, Tuple
import numpy as np
from scipy.stats import norm


# ─── Black-Scholes ────────────────────────────────────────────────────────

def bs_price(S: float, K: float, T_years: float, r: float,
             sigma: float, option_type: str = 'call') -> float:
    """Standard Black-Scholes price.
    sigma is annualized vol (decimal, e.g. 0.20 = 20%).
    """
    if T_years <= 0 or sigma <= 0:
        intrinsic = max(S - K, 0) if option_type == 'call' else max(K - S, 0)
        return float(intrinsic)
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T_years) / (sigma * np.sqrt(T_years))
    d2 = d1 - sigma * np.sqrt(T_years)
    if option_type == 'call':
        return float(S * norm.cdf(d1) - K * np.exp(-r * T_years) * norm.cdf(d2))
    else:  # put
        return float(K * np.exp(-r * T_years) * norm.cdf(-d2) - S * norm.cdf(-d1))


# ─── Empirical fair pricing (preferred) ───────────────────────────────────

def empirical_fair_chain(S_T_samples: np.ndarray, spot: float,
                         strike_step_pct: float = 1.0,
                         strike_range_pct: float = 12.0) -> dict:
    """Build a 'fair' option chain by directly integrating expected payoff
    over our empirical S_T distribution.

    No Black-Scholes assumption. No risk-neutral measure. No IV. The price of
    each option is exactly its mean payoff under our distribution.

    Note: with these prices, ANY strategy has E[P&L]=0 by construction.
    The point isn't to find positive-EV trades; it's to compare the SHAPE
    of P&L distributions across strategies.
    """
    n_strikes = int(2 * strike_range_pct / strike_step_pct) + 1
    strikes = np.array([
        spot * (1 + (i - n_strikes // 2) * strike_step_pct / 100)
        for i in range(n_strikes)
    ])
    S_T = np.asarray(S_T_samples, dtype=float)
    calls = np.array([np.mean(np.maximum(S_T - K, 0.0)) for K in strikes])
    puts  = np.array([np.mean(np.maximum(K - S_T, 0.0)) for K in strikes])
    # Implied vol equivalent (approximate) — just for display reference
    # Estimate ATM IV: ATM call price ≈ 0.4 × spot × σ × sqrt(T)
    # We don't know T here so we leave iv blank.
    return {
        'spot':        spot,
        'expiry_days': None,
        'strikes':     strikes,
        'calls':       calls,
        'puts':        puts,
        'iv_call':     np.full_like(strikes, np.nan),
        'iv_put':      np.full_like(strikes, np.nan),
        'source':      'empirical_fair',
    }


# ─── Synthetic Black-Scholes chain (kept for reference / IV mode) ────────

def build_synthetic_chain(spot: float, expiry_days: int,
                          base_iv: float, r: float = 0.045,
                          strike_step_pct: float = 1.0,
                          strike_range_pct: float = 12.0,
                          skew_slope: float = -0.4) -> dict:
    """Generate a synthetic option chain priced via Black-Scholes.

    Args:
        spot: current underlying price
        expiry_days: calendar days to expiry
        base_iv: ATM IV (decimal, e.g. 0.20)
        r: risk-free rate (decimal)
        strike_step_pct: spacing between strikes as % of spot
        strike_range_pct: total range each side of spot
        skew_slope: how much IV rises for OTM puts (per +10% OTM moneyness).
                    Default -0.4 means 10% OTM put has IV ~ base_iv * 1.04
                    (loose approximation of equity skew).

    Returns dict:
        { 'spot':..., 'expiry_days':..., 'T_years':...,
          'strikes': np.ndarray,
          'calls':  np.ndarray of prices,
          'puts':   np.ndarray of prices,
          'iv_call': np.ndarray,
          'iv_put':  np.ndarray,
        }
    """
    T_years = expiry_days / 365.0
    n_strikes = int(2 * strike_range_pct / strike_step_pct) + 1
    strikes = np.array([
        spot * (1 + (i - n_strikes // 2) * strike_step_pct / 100)
        for i in range(n_strikes)
    ])
    # IV per strike with simple linear skew (lower strike → higher IV for puts)
    moneyness = strikes / spot - 1.0   # 0 = ATM, negative = below spot
    iv_put  = np.clip(base_iv * (1 - skew_slope * moneyness), 0.05, 2.0)
    iv_call = np.clip(base_iv * (1 - 0.5 * skew_slope * moneyness), 0.05, 2.0)

    calls = np.array([bs_price(spot, K, T_years, r, iv, 'call')
                      for K, iv in zip(strikes, iv_call)])
    puts  = np.array([bs_price(spot, K, T_years, r, iv, 'put')
                      for K, iv in zip(strikes, iv_put)])
    return {
        'spot':        spot,
        'expiry_days': expiry_days,
        'T_years':     T_years,
        'r':           r,
        'base_iv':     base_iv,
        'strikes':     strikes,
        'calls':       calls,
        'puts':        puts,
        'iv_call':     iv_call,
        'iv_put':      iv_put,
        'source':      'synthetic_bs',
    }


# ─── Leg & payoff ─────────────────────────────────────────────────────────

@dataclass
class Leg:
    type: str           # 'call' | 'put'
    side: str           # 'long' | 'short'
    strike: float
    premium: float      # price paid (long) / received (short)  per share

    def cost(self) -> float:
        """Cost basis at entry. Positive = paid out, negative = received."""
        return self.premium if self.side == 'long' else -self.premium

    def payoff(self, S_T: np.ndarray) -> np.ndarray:
        """Payoff at expiry (per share, ignoring premium)."""
        if self.type == 'call':
            base = np.maximum(S_T - self.strike, 0)
        else:
            base = np.maximum(self.strike - S_T, 0)
        return base if self.side == 'long' else -base

    def __repr__(self) -> str:
        sign = '+' if self.side == 'long' else '-'
        return f'{sign}{self.type[0].upper()}{self.strike:.0f}@{self.premium:.2f}'


@dataclass
class Strategy:
    name: str
    legs: List[Leg] = field(default_factory=list)

    def cost(self) -> float:
        """Net cost (positive = debit, negative = credit)."""
        return sum(l.cost() for l in self.legs)

    def payoff(self, S_T: np.ndarray) -> np.ndarray:
        """Total payoff at expiry (per share, before subtracting cost)."""
        return sum(l.payoff(S_T) for l in self.legs)

    def pnl(self, S_T: np.ndarray) -> np.ndarray:
        """Total P&L per share."""
        return self.payoff(S_T) - self.cost()


# ─── Strategy templates ───────────────────────────────────────────────────

def _find_strike(strikes: np.ndarray, target: float) -> int:
    return int(np.argmin(np.abs(strikes - target)))


def make_long_call(chain, K):
    i = _find_strike(chain['strikes'], K)
    return Strategy(f'Long Call {chain["strikes"][i]:.0f}',
                    [Leg('call', 'long', chain['strikes'][i], chain['calls'][i])])


def make_short_call(chain, K):
    i = _find_strike(chain['strikes'], K)
    return Strategy(f'Short Call {chain["strikes"][i]:.0f}',
                    [Leg('call', 'short', chain['strikes'][i], chain['calls'][i])])


def make_long_put(chain, K):
    i = _find_strike(chain['strikes'], K)
    return Strategy(f'Long Put {chain["strikes"][i]:.0f}',
                    [Leg('put', 'long', chain['strikes'][i], chain['puts'][i])])


def make_short_put(chain, K):
    i = _find_strike(chain['strikes'], K)
    return Strategy(f'Short Put {chain["strikes"][i]:.0f}',
                    [Leg('put', 'short', chain['strikes'][i], chain['puts'][i])])


def make_bull_call_spread(chain, K_long, K_short):
    iL = _find_strike(chain['strikes'], K_long)
    iS = _find_strike(chain['strikes'], K_short)
    return Strategy(
        f'Bull Call Spread {chain["strikes"][iL]:.0f}/{chain["strikes"][iS]:.0f}',
        [Leg('call', 'long',  chain['strikes'][iL], chain['calls'][iL]),
         Leg('call', 'short', chain['strikes'][iS], chain['calls'][iS])])


def make_bear_put_spread(chain, K_long, K_short):
    iL = _find_strike(chain['strikes'], K_long)
    iS = _find_strike(chain['strikes'], K_short)
    return Strategy(
        f'Bear Put Spread {chain["strikes"][iL]:.0f}/{chain["strikes"][iS]:.0f}',
        [Leg('put', 'long',  chain['strikes'][iL], chain['puts'][iL]),
         Leg('put', 'short', chain['strikes'][iS], chain['puts'][iS])])


def make_bull_put_spread(chain, K_short, K_long):
    """Credit spread (sell put higher, buy put lower)."""
    iS = _find_strike(chain['strikes'], K_short)
    iL = _find_strike(chain['strikes'], K_long)
    return Strategy(
        f'Bull Put Spread {chain["strikes"][iL]:.0f}/{chain["strikes"][iS]:.0f}',
        [Leg('put', 'short', chain['strikes'][iS], chain['puts'][iS]),
         Leg('put', 'long',  chain['strikes'][iL], chain['puts'][iL])])


def make_bear_call_spread(chain, K_short, K_long):
    """Credit spread (sell call lower, buy call higher)."""
    iS = _find_strike(chain['strikes'], K_short)
    iL = _find_strike(chain['strikes'], K_long)
    return Strategy(
        f'Bear Call Spread {chain["strikes"][iS]:.0f}/{chain["strikes"][iL]:.0f}',
        [Leg('call', 'short', chain['strikes'][iS], chain['calls'][iS]),
         Leg('call', 'long',  chain['strikes'][iL], chain['calls'][iL])])


def make_long_straddle(chain, K):
    i = _find_strike(chain['strikes'], K)
    return Strategy(
        f'Long Straddle {chain["strikes"][i]:.0f}',
        [Leg('call', 'long', chain['strikes'][i], chain['calls'][i]),
         Leg('put',  'long', chain['strikes'][i], chain['puts'][i])])


def make_short_straddle(chain, K):
    i = _find_strike(chain['strikes'], K)
    return Strategy(
        f'Short Straddle {chain["strikes"][i]:.0f}',
        [Leg('call', 'short', chain['strikes'][i], chain['calls'][i]),
         Leg('put',  'short', chain['strikes'][i], chain['puts'][i])])


def make_long_strangle(chain, K_put, K_call):
    iP = _find_strike(chain['strikes'], K_put)
    iC = _find_strike(chain['strikes'], K_call)
    return Strategy(
        f'Long Strangle {chain["strikes"][iP]:.0f}/{chain["strikes"][iC]:.0f}',
        [Leg('put',  'long', chain['strikes'][iP], chain['puts'][iP]),
         Leg('call', 'long', chain['strikes'][iC], chain['calls'][iC])])


def make_short_strangle(chain, K_put, K_call):
    iP = _find_strike(chain['strikes'], K_put)
    iC = _find_strike(chain['strikes'], K_call)
    return Strategy(
        f'Short Strangle {chain["strikes"][iP]:.0f}/{chain["strikes"][iC]:.0f}',
        [Leg('put',  'short', chain['strikes'][iP], chain['puts'][iP]),
         Leg('call', 'short', chain['strikes'][iC], chain['calls'][iC])])


def make_iron_condor(chain, K_put_long, K_put_short, K_call_short, K_call_long):
    """Sell put spread + sell call spread (credit, range-bound bet)."""
    return Strategy(
        f'Iron Condor {K_put_long:.0f}/{K_put_short:.0f}/{K_call_short:.0f}/{K_call_long:.0f}',
        make_bull_put_spread(chain, K_put_short, K_put_long).legs +
        make_bear_call_spread(chain, K_call_short, K_call_long).legs
    )


def make_iron_butterfly(chain, K_atm, K_wing_put, K_wing_call):
    return Strategy(
        f'Iron Butterfly {K_wing_put:.0f}/{K_atm:.0f}/{K_wing_call:.0f}',
        make_bull_put_spread(chain, K_atm, K_wing_put).legs +
        make_bear_call_spread(chain, K_atm, K_wing_call).legs
    )


__all__ = [
    'bs_price', 'build_synthetic_chain', 'empirical_fair_chain',
    'Leg', 'Strategy',
    'make_long_call', 'make_short_call', 'make_long_put', 'make_short_put',
    'make_bull_call_spread', 'make_bear_put_spread',
    'make_bull_put_spread', 'make_bear_call_spread',
    'make_long_straddle', 'make_short_straddle',
    'make_long_strangle', 'make_short_strangle',
    'make_iron_condor', 'make_iron_butterfly',
]
