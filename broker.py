"""
Algorithmic Trading Bot – Version 1 (paper-trade friendly)
Author: You + ChatGPT

What this does
--------------
- Pulls fundamentals and prices with yfinance (free)
- Computes rough ROIC, Debt/Equity, and 5-year Revenue CAGR
- Scores and ranks stocks using your value/growth rules
- Builds a Top N list and a simple equal-weight portfolio suggestion
- (Optional) Backtests monthly rebalanced performance vs SPY

How you should use it
---------------------
1) Install deps once:
   pip install --upgrade yfinance pandas numpy matplotlib

2) Run the script:
   python algo_bot_v1.py --universe sp500 --top 15 --rebalance M --years 5

3) Make manual trades in Robinhood based on the recommendations (no auto-trading).

Notes
-----
- ROIC is approximated with publicly available statements and may not match GAAP/analyst figures exactly.
- You can swap the universe to a custom CSV of tickers (see --universe help).
- Backtest is *educational* – it ignores slippage, taxes, borrow fees, etc.
"""

from __future__ import annotations
import argparse
import math
import sys
from dataclasses import dataclass
from typing import List, Dict, Tuple

import numpy as np
import pandas as pd
import yfinance as yf

# ---------------------------
# Utility helpers
# ---------------------------

def safe_get(series: pd.Series, key: str, default=np.nan):
    """Return series[key] if present, else default."""
    try:
        return series.get(key, default)
    except Exception:
        return default


def CAGR(start: float, end: float, years: float) -> float:
    if start <= 0 or end <= 0 or years <= 0:
        return np.nan
    return (end / start) ** (1.0 / years) - 1.0


def pct(x):
    return 100.0 * x

# ---------------------------
# Data acquisition
# ---------------------------

def get_universe(kind: str) -> List[str]:
    """Return a list of tickers for the chosen universe.
    kind can be:
      - 'sp500' (scrapes Wikipedia)
      - path to a CSV file with a 'ticker' column
      - comma-separated tickers, e.g. 'AAPL,MSFT,GOOGL'
    """
    kind = kind.strip()
    if kind.lower() == "sp500":
        try:
            tables = pd.read_html("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")
            df = tables[0]
            tickers = df["Symbol"].astype(str).str.replace(".", "-", regex=False).tolist()
            return tickers
        except Exception as e:
            print(f"Failed to fetch S&P 500 list: {e}\nProvide a CSV with a 'ticker' column or a comma list instead.")
            sys.exit(1)
    elif "," in kind:
        return [t.strip().upper() for t in kind.split(",") if t.strip()]
    else:
        # Assume CSV path
        df = pd.read_csv(kind)
        col = None
        for c in df.columns:
            if c.lower() in {"ticker", "tickers", "symbol", "symbols"}:
                col = c
                break
        if not col:
            raise ValueError("CSV must contain a 'ticker' column")
        return df[col].astype(str).str.upper().tolist()


@dataclass
class Fundamentals:
    ticker: str
    roic: float
    de_ratio: float
    rev_cagr_5y: float
    mcap: float
    fcf_ttm: float


def fetch_fundamentals(ticker: str) -> Fundamentals | None:
    """Pulls a subset of fundamental data via yfinance and computes metrics.
    ROIC approximation: NOPAT / Invested Capital
      - NOPAT ~ EBIT * (1 - tax_rate)
      - Invested Capital ~ Total Debt + Total Equity - Cash & Equivalents
    D/E ratio: Total Debt / Total Equity
    Revenue CAGR: 5y from annual income statements
    """
    try:
        tk = yf.Ticker(ticker)

        # Balance sheet, income statement, cashflow (prefer annual)
        # yfinance returns DataFrames with columns as dates (most recent first)
        bs = None
        inc = None
        cf = None
        info = tk.fast_info if hasattr(tk, "fast_info") else {}
        try:
            bs = tk.balance_sheet
        except Exception:
            bs = None
        try:
            inc = tk.income_stmt
        except Exception:
            try:
                inc = tk.financials
            except Exception:
                inc = None
        try:
            cf = tk.cashflow
        except Exception:
            cf = None

        if bs is None or inc is None:
            return None

        # Use the most recent annual column when available
        def latest_col(df: pd.DataFrame):
            return df.columns[0] if isinstance(df, pd.DataFrame) and df.shape[1] > 0 else None

        col_latest_bs = latest_col(bs)
        col_latest_inc = latest_col(inc)
        col_latest_cf = latest_col(cf) if cf is not None else None

        bs_latest = bs[col_latest_bs] if col_latest_bs is not None else pd.Series(dtype=float)
        inc_latest = inc[col_latest_inc] if col_latest_inc is not None else pd.Series(dtype=float)
        cf_latest = cf[col_latest_cf] if (cf is not None and col_latest_cf is not None) else pd.Series(dtype=float)

        # Components for ROIC
        ebit = safe_get(inc_latest, "EBIT", np.nan)
        if np.isnan(ebit):
            ebit = safe_get(inc_latest, "Operating Income", np.nan)

        income_tax = safe_get(inc_latest, "Income Tax Expense", np.nan)
        pretax = safe_get(inc_latest, "Income Before Tax", np.nan)
        tax_rate = (income_tax / pretax) if (isinstance(income_tax, (int, float, np.floating)) and isinstance(pretax, (int, float, np.floating)) and pretax not in (0, np.nan)) else 0.21  # fallback ~US corporate
        nopat = ebit * (1 - tax_rate) if not np.isnan(ebit) else np.nan

        total_debt = safe_get(bs_latest, "Total Debt", np.nan)
        if np.isnan(total_debt):
            st_debt = safe_get(bs_latest, "Short Long Term Debt", 0.0)
            lt_debt = safe_get(bs_latest, "Long Term Debt", 0.0)
            total_debt = st_debt + lt_debt
        equity = safe_get(bs_latest, "Stockholders Equity", np.nan)
        cash = safe_get(bs_latest, "Cash And Cash Equivalents", np.nan)
        invested_capital = np.nan
        if not np.isnan(total_debt) and not np.isnan(equity) and not np.isnan(cash):
            invested_capital = total_debt + equity - cash
        roic = (nopat / invested_capital) if (isinstance(nopat, (int, float, np.floating)) and isinstance(invested_capital, (int, float, np.floating)) and invested_capital not in (0, np.nan)) else np.nan

        # Debt/Equity
        de_ratio = (total_debt / equity) if (isinstance(total_debt, (int, float, np.floating)) and isinstance(equity, (int, float, np.floating)) and equity not in (0, np.nan)) else np.nan

        # Market cap
        mcap = np.nan
        try:
            mcap = float(getattr(tk, "info", {}).get("marketCap") or getattr(tk, "fast_info", {}).get("market_cap"))
        except Exception:
            pass

        # Free cash flow (ttm approx)
        fcf_ttm = safe_get(cf_latest, "Free Cash Flow", np.nan)
        if np.isnan(fcf_ttm):
            cfo = safe_get(cf_latest, "Total Cash From Operating Activities", np.nan)
            capex = safe_get(cf_latest, "Capital Expenditures", np.nan)
            if not np.isnan(cfo) and not np.isnan(capex):
                fcf_ttm = cfo - capex

        # Revenue CAGR 5y
        rev_cagr_5y = np.nan
        try:
            inc_annual = tk.income_stmt
            if inc_annual is not None and inc_annual.shape[1] >= 5:
                rev = inc_annual.loc["Total Revenue"].sort_index()
                start, end = float(rev.iloc[0]), float(rev.iloc[-1])
                years = (rev.index[-1].year - rev.index[0].year) or (len(rev) - 1)
                years = max(years, 1)
                rev_cagr_5y = CAGR(start, end, years)
        except Exception:
            pass

        return Fundamentals(
            ticker=ticker,
            roic=roic,
            de_ratio=de_ratio,
            rev_cagr_5y=rev_cagr_5y,
            mcap=mcap,
            fcf_ttm=fcf_ttm,
        )
    except Exception:
        return None


# ---------------------------
# Scoring model
# ---------------------------

def score_row(row: pd.Series, weights: Dict[str, float]) -> float:
    """Compute a composite score with clipping/normalization."""
    roic = row.get("roic", np.nan)
    de = row.get("de_ratio", np.nan)
    cagr = row.get("rev_cagr_5y", np.nan)

    # Clip and transform
    roic_s = np.tanh(np.nan_to_num(roic, nan=0.0) * 2)  # emphasize 0-50% range
    de_s = -np.tanh(np.nan_to_num(de, nan=2.0))         # lower D/E is better
    cagr_s = np.tanh(np.nan_to_num(cagr, nan=0.0) * 2)

    return (
        weights.get("roic", 0.5) * roic_s +
        weights.get("de", 0.2) * de_s +
        weights.get("cagr", 0.3) * cagr_s
    )


# ---------------------------
# Backtest (simple, monthly rebalance)
# ---------------------------

def monthly_rebalance_backtest(tickers: List[str], start: str, end: str, top_n: int, weights_cfg: Dict[str, float],
                               universe_kind: str, rebalance: str = "M") -> Tuple[pd.DataFrame, pd.Series]:
    """Builds a dynamic portfolio that re-screens each rebalance period and compares to SPY."""
    # Price data for universe
    px = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=False)["Close"]
    if isinstance(px, pd.Series):
        px = px.to_frame()

    # Monthly endpoints
    monthly = px.resample("M").last()

    # Prepare fundamentals snapshot per period (naively reuse latest values due to API limits)
    # For educational purposes we re-score on each rebalance using the *current* fundamentals when backtest runs.
    # Realistically, you would store historical fundamentals by period.

    periods = monthly.index
    port_vals = []
    weights_hist = []
    last_prices = None

    for i, dt in enumerate(periods):
        last_dt = periods[i-1] if i > 0 else None
        if last_dt is None:
            last_prices = px.loc[:dt].iloc[-1]
            port_vals.append(1.0)
            weights_hist.append(pd.Series(dtype=float))
            continue

        # Re-score universe using (static) fundamentals fetched now
        funs = []
        for t in tickers:
            f = fetch_fundamentals(t)
            if f is None:
                continue
            funs.append(vars(f))
        if not funs:
            raise RuntimeError("No fundamentals fetched – try fewer tickers or a different universe.")
        df = pd.DataFrame(funs).drop_duplicates("ticker").set_index("ticker")
        df["score"] = df.apply(lambda r: score_row(r, weights_cfg), axis=1)
        picks = df.sort_values("score", ascending=False).head(top_n).index.tolist()

        # Compute portfolio value: equal-weight among picks
        sub_px = px[picks]
        # returns for the period (last_dt, dt]
        p0 = sub_px.loc[last_dt]
        p1 = sub_px.loc[dt]
        rets = (p1 / p0) - 1.0
        period_ret = rets.mean(skipna=True)

        # accumulate
        port_vals.append(port_vals[-1] * (1 + period_ret))
        w = pd.Series(1.0 / len(picks), index=picks)
        weights_hist.append(w)

    port = pd.Series(port_vals, index=periods, name="Portfolio")

    # Benchmark
    spy = yf.download(["SPY"], start=start, end=end, auto_adjust=True, progress=False)["Close"].resample("M").last()
    spy = spy / spy.iloc[0]
    bench = spy.rename("SPY")

    df = pd.concat([port, bench], axis=1).dropna()
    return df, weights_hist[-1] if weights_hist else pd.Series(dtype=float)


# ---------------------------
# Main screening pipeline
# ---------------------------

def screen(universe_kind: str, max_tickers: int, weights_cfg: Dict[str, float]) -> pd.DataFrame:
    tickers = get_universe(universe_kind)
    if max_tickers:
        tickers = tickers[:max_tickers]

    rows = []
    for t in tickers:
        f = fetch_fundamentals(t)
        if f is None:
            continue
        rows.append(vars(f))

    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError("No data fetched. Try a smaller universe or check your internet connection.")

    # Basic sanity filters
    df = df[df["mcap"].fillna(0) > 2_000_000_000]  # > $2B
    df = df[df["fcf_ttm"].fillna(-1) > 0]          # positive FCF

    # Compute score
    df["score"] = df.apply(lambda r: score_row(r, weights_cfg), axis=1)

    # Sort
    df = df.sort_values("score", ascending=False)
    return df


def main():
    p = argparse.ArgumentParser(description="Rule-based stock screener + simple backtest (Robinhood-friendly)")
    p.add_argument("--universe", default="sp500", help="sp500 | path/to.csv | comma,list,of,tickers")
    p.add_argument("--limit", type=int, default=200, help="Max tickers to scan from universe (to keep it speedy)")
    p.add_argument("--top", type=int, default=10, help="How many top names to output")
    p.add_argument("--years", type=int, default=5, help="Backtest years")
    p.add_argument("--rebalance", default="M", choices=["M"], help="Rebalance frequency (M = monthly)")
    p.add_argument("--no_backtest", action="store_true", help="Skip backtest, only screen")
    p.add_argument("--weights", default="roic=0.5,de=0.2,cagr=0.3", help="Weights like roic=0.5,de=0.2,cagr=0.3")

    args = p.parse_args()

    # Parse weights
    weights_cfg = {k: float(v) for k, v in (kv.split("=") for kv in args.weights.split(","))}

    print("\n=== Screening Universe ===")
    df = screen(args.universe, args.limit, weights_cfg)

    print("Top candidates (composite score):\n")
    cols = ["ticker", "score", "roic", "de_ratio", "rev_cagr_5y", "mcap", "fcf_ttm"]
    out = df.reset_index(drop=True).copy()
    out = out[cols]
    out["roic"] = out["roic"].apply(lambda x: f"{pct(x):.2f}%" if pd.notna(x) else "–")
    out["rev_cagr_5y"] = out["rev_cagr_5y"].apply(lambda x: f"{pct(x):.2f}%" if pd.notna(x) else "–")
    out["de_ratio"] = out["de_ratio"].apply(lambda x: f"{x:.2f}" if pd.notna(x) else "–")
    out["mcap"] = out["mcap"].apply(lambda x: f"${x/1e9:.1f}B" if pd.notna(x) else "–")
    out["fcf_ttm"] = out["fcf_ttm"].apply(lambda x: f"${x/1e9:.2f}B" if pd.notna(x) else "–")

    print(out.head(args.top).to_string(index=False))

    if args.no_backtest:
        return

    print("\n=== Simple Backtest ===")
    end = pd.Timestamp.today().normalize()
    start = end - pd.DateOffset(years=args.years)

    tickers = get_universe(args.universe)[:args.limit]
    perf, last_weights = monthly_rebalance_backtest(tickers, start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d"),
                                                    args.top, weights_cfg, args.universe, args.rebalance)
    # Performance stats
    port = perf["Portfolio"].dropna()
    spy = perf["SPY"].dropna()

    port_ret = port.iloc[-1] - 1.0
    spy_ret = spy.iloc[-1] - 1.0
    port_cagr = CAGR(port.iloc[0], port.iloc[-1], args.years)
    spy_cagr = CAGR(spy.iloc[0], spy.iloc[-1], args.years)

    print(f"Portfolio total return ({args.years}y): {pct(port_ret):.2f}%  |  CAGR: {pct(port_cagr):.2f}%")
    print(f"SPY total return ({args.years}y):       {pct(spy_ret):.2f}%  |  CAGR: {pct(spy_cagr):.2f}%")

    # Show latest weights suggestion
    if not last_weights.empty:
        print("\nSuggested equal-weight allocation (last rebalance):")
        alloc = (last_weights / last_weights.sum()).sort_values(ascending=False)
        for t, w in alloc.items():
            print(f"  {t}: {pct(w):.1f}%")


if __name__ == "__main__":
    main()