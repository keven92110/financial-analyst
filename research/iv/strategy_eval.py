"""Evaluate and rank option strategies under our empirical distribution.

Given:
  - Empirical samples of fwd log-returns (from our model's prediction)
  - Spot price S0
  - Synthetic option chain at the chosen expiry

We compute, for each candidate strategy, under the empirical distribution of S_T:
  - Expected P&L (E[P&L])
  - Probability of profit (PoP)
  - Standard deviation of P&L
  - Max gain / max loss
  - Breakeven points
  - Sharpe-like ratio E[P&L] / std(P&L)
"""
from __future__ import annotations
from typing import List, Dict
import numpy as np

from iv.strategy import (
    Strategy, build_synthetic_chain, empirical_fair_chain,
    make_long_call, make_short_call, make_long_put, make_short_put,
    make_bull_call_spread, make_bear_put_spread,
    make_bull_put_spread, make_bear_call_spread,
    make_long_straddle, make_short_straddle,
    make_long_strangle, make_short_strangle,
    make_iron_condor, make_iron_butterfly,
)


# ─── Distribution helpers ─────────────────────────────────────────────────

def samples_to_prices(log_returns: np.ndarray, S0: float) -> np.ndarray:
    """Convert log-return samples to terminal price samples."""
    return S0 * np.exp(log_returns)


# ─── Single-strategy evaluation ───────────────────────────────────────────

def evaluate(strat: Strategy, S_T: np.ndarray) -> dict:
    """Compute all summary stats for `strat` over price samples `S_T`."""
    if len(S_T) == 0:
        return {'name': strat.name, 'n': 0}
    pnl = strat.pnl(S_T)
    cost = strat.cost()
    e_pnl = float(pnl.mean())
    std_pnl = float(pnl.std())
    pop = float((pnl > 0).mean() * 100)

    # E[P&L] decomposition: contribution from winning vs losing scenarios
    # e_upside   = (1/N) * sum of pnl where pnl > 0   → expected gain contribution
    # e_downside = (1/N) * sum of -pnl where pnl < 0  → expected loss contribution (magnitude)
    # By identity: e_upside - e_downside = e_pnl   → both equal under fair pricing
    e_upside   = float(pnl[pnl > 0].sum() / len(pnl)) if (pnl > 0).any() else 0.0
    e_downside = float(-pnl[pnl < 0].sum() / len(pnl)) if (pnl < 0).any() else 0.0

    # Breakeven detection
    grid = np.linspace(S_T.min(), S_T.max(), 200)
    pnl_grid = strat.pnl(grid)
    sign_changes = np.where(np.diff(np.sign(pnl_grid)) != 0)[0]
    breakevens = [float((grid[i] + grid[i+1]) / 2) for i in sign_changes]

    return {
        'name':         strat.name,
        'cost':         float(cost),    # debit positive, credit negative
        'e_pnl':        e_pnl,
        'std_pnl':      std_pnl,
        'sharpe':       float(e_pnl / std_pnl) if std_pnl > 0 else 0.0,
        'pop':          pop,
        'max_gain':     float(pnl.max()),
        'max_loss':     float(pnl.min()),
        'e_upside':     e_upside,
        'e_downside':   e_downside,
        'p10':          float(np.percentile(pnl, 10)),
        'p90':          float(np.percentile(pnl, 90)),
        'breakevens':   breakevens,
        'n_samples':    int(len(S_T)),
        'strategy':     strat,
    }


def expected_contribution_curve(strat: Strategy, S_T: np.ndarray,
                                n_grid: int = 300, n_bins: int = 80) -> dict:
    """Return (S_grid, pnl_grid, density, contribution) for the chart.

    contribution = pnl(S_T) × pdf(S_T)
    Area under positive part = expected gain contribution
    Area under negative part = expected loss contribution (negative number)
    Total area = E[P&L]
    """
    s_lo = float(np.percentile(S_T, 0.5)) * 0.97
    s_hi = float(np.percentile(S_T, 99.5)) * 1.03
    grid = np.linspace(s_lo, s_hi, n_grid)
    # Estimate density from histogram (smoother than KDE for our skewed data)
    hist, edges = np.histogram(S_T, bins=n_bins, density=True)
    centers = (edges[:-1] + edges[1:]) / 2
    density = np.interp(grid, centers, hist, left=0, right=0)
    pnl_grid = strat.pnl(grid)
    contribution = pnl_grid * density
    return {
        'grid':         grid,
        'pnl_grid':     pnl_grid,
        'density':      density,
        'contribution': contribution,
    }


# ─── Strategy generation (search space) ───────────────────────────────────

def generate_candidates(chain, S0: float,
                        sigma_log: float = None,
                        primary_sigma_mult: float = 1.0,
                        sigma_mults: list = None) -> List[Strategy]:
    """Enumerate candidate strategies using sigma-based strike selection.

    Args:
        chain: option chain (from empirical_fair_chain or build_synthetic_chain)
        S0: spot price
        sigma_log: std of log-returns over horizon (decimal). If None, uses
                   chain's range to estimate.
        primary_sigma_mult: user-chosen primary sigma multiplier (default 1.0).
                           Strategies will use this and a few variants around it.
        sigma_mults: optional list of all sigma multipliers to use. If None,
                     auto-derives from primary_sigma_mult: [0.5×, 1×, 1.5×, 2×]
    """
    if sigma_log is None or sigma_log <= 0:
        sigma_log = 0.05   # fallback ~5% (only if upstream forgot to pass)

    if sigma_mults is None:
        # Build a grid around the user's primary multiplier
        sigma_mults = [0.5 * primary_sigma_mult,
                       1.0 * primary_sigma_mult,
                       1.5 * primary_sigma_mult,
                       2.0 * primary_sigma_mult]

    def K_at(m: float) -> float:
        """Strike at m sigma above spot (m can be negative)."""
        return S0 * float(np.exp(m * sigma_log))

    cands: List[Strategy] = []

    # ── Single legs at each ±k×σ
    for m in sigma_mults:
        cands.append(make_long_call(chain, K_at(m)))
        cands.append(make_short_call(chain, K_at(m)))
        cands.append(make_long_put(chain, K_at(-m)))
        cands.append(make_short_put(chain, K_at(-m)))

    # ── ATM straddle
    cands.append(make_long_straddle(chain, S0))
    cands.append(make_short_straddle(chain, S0))

    # ── Strangle at each ±k×σ
    for m in sigma_mults:
        cands.append(make_long_strangle(chain, K_at(-m), K_at(m)))
        cands.append(make_short_strangle(chain, K_at(-m), K_at(m)))

    # ── Vertical spreads between adjacent sigma levels
    sorted_mults = sorted(set(sigma_mults))
    for i in range(len(sorted_mults) - 1):
        m_inner = sorted_mults[i]
        m_outer = sorted_mults[i + 1]
        # Bull put spread (credit): short put at -inner, long put at -outer
        cands.append(make_bull_put_spread(chain, K_at(-m_inner), K_at(-m_outer)))
        # Bear call spread (credit): short call at +inner, long call at +outer
        cands.append(make_bear_call_spread(chain, K_at(m_inner), K_at(m_outer)))
        # Bull call spread (debit): long call at +inner, short call at +outer
        cands.append(make_bull_call_spread(chain, K_at(m_inner), K_at(m_outer)))
        # Bear put spread (debit): long put at -inner, short put at -outer
        cands.append(make_bear_put_spread(chain, K_at(-m_inner), K_at(-m_outer)))

    # ── Iron condors using inner & outer sigma levels
    if len(sorted_mults) >= 2:
        # primary condor uses the user's primary level as inner, larger as outer
        inner = primary_sigma_mult
        outer_options = [m for m in sorted_mults if m > inner]
        if not outer_options:
            outer_options = [inner * 2.0]
        for outer in outer_options:
            cands.append(make_iron_condor(
                chain, K_at(-outer), K_at(-inner),
                K_at(inner), K_at(outer)))

    # ── Iron butterflies (ATM body + wings at sigma levels)
    for wing in sigma_mults:
        cands.append(make_iron_butterfly(
            chain, S0, K_at(-wing), K_at(wing)))

    return cands


# ─── Top-level entry point ────────────────────────────────────────────────

def find_best_strategies(log_return_samples: np.ndarray,
                         S0: float,
                         expiry_days: int = None,
                         base_iv: float = None,
                         pricing: str = 'empirical',
                         primary_sigma_mult: float = 1.0,
                         top_k: int = 12,
                         metric: str = 'pop') -> dict:
    """Build chain → enumerate strategies → evaluate → return ranked list.

    pricing:
      'empirical' (default) — option prices = expected payoff under our
                              distribution. EV is 0 for every strategy by
                              construction; ranking is by P&L *shape*.
      'bs'                  — Black-Scholes pricing using base_iv. Strategies
                              can have nonzero EV if our distribution differs
                              from log-normal.

    metric:
      'pop'    — probability of profit (good when EV is 0 for all strategies)
      'sharpe' — E[P&L] / std(P&L) (more meaningful under BS pricing)
      'e_pnl'  — raw expected P&L (mostly useful under BS pricing)
    """
    S_T = samples_to_prices(log_return_samples, S0)
    sigma_log = float(np.std(log_return_samples))   # std of log-returns over horizon

    if pricing == 'empirical':
        # Wider strike grid for sigma-based selection (need ±2.5σ usable)
        chain = empirical_fair_chain(S_T, S0,
                                     strike_step_pct=0.5,
                                     strike_range_pct=20.0)
    else:
        if base_iv is None or expiry_days is None:
            raise ValueError('base_iv and expiry_days required for BS pricing')
        chain = build_synthetic_chain(S0, expiry_days, base_iv,
                                      strike_step_pct=0.5,
                                      strike_range_pct=20.0)

    candidates = generate_candidates(chain, S0,
                                     sigma_log=sigma_log,
                                     primary_sigma_mult=primary_sigma_mult)

    results = []
    for strat in candidates:
        res = evaluate(strat, S_T)
        res['e_pnl_pct_of_spot'] = res['e_pnl'] / S0 * 100
        results.append(res)

    key_map = {'e_pnl': 'e_pnl', 'sharpe': 'sharpe', 'pop': 'pop'}
    sort_key = key_map.get(metric, 'pop')
    results.sort(key=lambda r: r[sort_key], reverse=True)
    return {'top': results[:top_k],
            'all': results,
            'chain': chain,
            'S_T':   S_T,
            'S0':    S0,
            'sigma_log':            sigma_log,
            'primary_sigma_mult':   primary_sigma_mult,
            'pricing': pricing}


__all__ = [
    'samples_to_prices',
    'evaluate', 'expected_contribution_curve',
    'generate_candidates', 'find_best_strategies',
]


# ─── CLI test ────────────────────────────────────────────────────────────

if __name__ == '__main__':
    import sys, io, os
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))   # research/

    import predict
    print('Loading today signals (SPX)...')
    data = predict.get_today_signals()
    spx_entry = data['indices'].get('SPX', {})
    cls = spx_entry.get('classification') or {}
    preds = spx_entry.get('predictions', {})

    fwd_5d_vals  = preds.get('cat_d', {}).get('fwd_5d',  {}).get('vals')
    fwd_20d_vals = preds.get('cat_d', {}).get('fwd_20d', {}).get('vals')

    S0 = cls.get('last_close', 7000)
    print(f'\nSPX spot: {S0:.2f}')
    print(f'today cat_d: {cls.get("cat_d")}')

    for hkey in ['fwd_5d', 'fwd_20d']:
        vals = preds.get('cat_d', {}).get(hkey, {}).get('vals')
        if vals is None or len(vals) == 0:
            continue
        print(f'\n{"="*88}')
        print(f'STRATEGY SEARCH (empirical fair pricing) · horizon={hkey}')
        print(f'{"="*88}')
        out = find_best_strategies(np.asarray(vals), S0,
                                   pricing='empirical',
                                   top_k=15, metric='pop')
        print(f"{'rank':>4s}  {'name':38s}  {'cost':>7s}  {'E[P&L]':>8s}  "
              f"{'Sharpe':>6s}  {'PoP%':>5s}  {'maxGain':>8s}  {'maxLoss':>9s}")
        for i, r in enumerate(out['top'], 1):
            print(f"{i:>4d}  {r['name']:38s}  {r['cost']:+7.2f}  "
                  f"{r['e_pnl']:+8.2f}  {r['sharpe']:+6.2f}  "
                  f"{r['pop']:5.1f}  {r['max_gain']:+8.2f}  {r['max_loss']:+9.2f}")
