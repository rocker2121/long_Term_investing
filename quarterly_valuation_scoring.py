#!/usr/bin/env python3
# Quarterly Valuation Scoring (Sector-aware)
# Author: ChatGPT (for ANGAD ARORA)
# License: MIT
#
# What it does:
# - Loads a ticker universe
# - Fetches SEC XBRL facts (TTM revenue, net income, EBITDA proxy, etc.)
# - Fetches Alpha Vantage price (near now_date) and Sector via OVERVIEW
# - Computes valuation ratios (P/E, EV/EBITDA, P/B, P/S, FCF yield)
# - Adds quality/risk metrics (ROE, EBITDA margin, Net debt/EBITDA, Interest coverage)
# - Produces a 1..10 undervaluation score (1 = undervalued, 10 = overvalued), sector-aware
#
# Usage example:
#   python3 quarterly_valuation_scoring.py \
#     --universe_csv sp500.csv \
#     --now_date 2025-06-30 \
#     --price_vendor alpha_vantage \
#     --price_api_key 'YOUR_PREMIUM_KEY' \
#     --outdir ./val_output \
#     --av_throttle_sec 1.0 --av_max_retries 3 \
#     --sectors_csv sectors.csv   # optional; overrides AV sector if provided
#
# Notes:
# - Set a real SEC user-agent email below (SEC_UA)
# - This script focuses on rule-based scoring; no regression is trained
# - If a denominator is <=0, the affected metric is set to NaN (excluded from composite)
# - Sector awareness uses both absolute and within-sector deciles + sector-wide relief
# - You can run this once per quarter with an appropriate --now_date (e.g., quarter end)

import argparse
import csv
import datetime as dt
import gzip
import json
import math
import os
import sys
import time
import urllib.parse
import urllib.request
from functools import lru_cache
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# -----------------------------
# Config & Constants
# -----------------------------
SEC_UA = "QuarterlyValuation/1.0 (youremail@example.com)"  # <-- replace with your email
SEC_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
SEC_FACTS_URL_TMPL = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"

US_GAAP = "us-gaap"
TAGSETS = {
    "revenue": ["RevenueFromContractWithCustomerExcludingAssessedTax", "Revenues", "SalesRevenueNet"],
    "net_income": ["NetIncomeLoss"],
    "operating_income": ["OperatingIncomeLoss"],
    "dep_amort": ["DepreciationAndAmortization", "Depreciation"],
    "gross_profit": ["GrossProfit"],
    "assets": ["Assets"],
    "equity": ["StockholdersEquity", "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest"],
    "debt_longterm": ["LongTermDebt", "LongTermDebtNoncurrent"],
    "debt_current": ["LongTermDebtCurrent"],
    "short_term_borrow": ["ShortTermBorrowings"],
    "cash_and_equiv": ["CashAndCashEquivalentsAtCarryingValue"],
    "assets_current": ["AssetsCurrent"],
    "liabilities_current": ["LiabilitiesCurrent"],
    "interest_expense": ["InterestExpense"],
    "capex": ["PaymentsToAcquirePropertyPlantAndEquipment"],
    # For EPS / market cap:
    "shares_diluted": ["WeightedAverageNumberOfDilutedSharesOutstanding"],
    # For FCF:
    "operating_cf": ["NetCashProvidedByUsedInOperatingActivities",
                     "NetCashProvidedByUsedInOperatingActivitiesContinuingOperations"],
}

# -----------------------------
# HTTP helpers
# -----------------------------
def http_get(url: str, ua: str, timeout: int = 45, retries: int = 4, backoff: float = 2.0) -> bytes:
    last_err = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": ua, "Accept-Encoding": "gzip, deflate"})
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return r.read()
        except Exception as e:
            last_err = e
            time.sleep(backoff ** attempt)
    raise last_err

def to_date(s: str) -> Optional[dt.date]:
    try:
        return dt.date.fromisoformat(s[:10])
    except Exception:
        return None

# -----------------------------
# SEC helpers
# -----------------------------
def last_quarters(values, within_days=600, want=4):
    rows, today = [], dt.date.today()
    for f in values:
        end = to_date(f.get("end", ""))
        if not end or f.get("fp") not in ("Q1", "Q2", "Q3", "Q4") or (today - end).days > within_days:
            continue
        val = f.get("val")
        if isinstance(val, (int, float)):
            rows.append((end, float(val)))
    rows.sort(key=lambda x: x[0], reverse=True)
    return rows[:want]

def pick_last4_quarterly(facts: dict, taglist: List[str]):
    gaap = facts.get("facts", {}).get(US_GAAP, {})
    for t in taglist:
        node = gaap.get(t, {})
        for _, arr in node.get("units", {}).items():
            q = last_quarters(arr, within_days=600, want=4)
            if q:
                return q
    return []

def latest_value(facts: dict, taglist: List[str]):
    gaap = facts.get("facts", {}).get(US_GAAP, {})
    best = None
    for t in taglist:
        node = gaap.get(t, {})
        for _, arr in node.get("units", {}).items():
            for f in arr:
                end, val = to_date(f.get("end","")), f.get("val")
                if end and isinstance(val, (int, float)):
                    if (best is None) or (end > best[0]):
                        best = (end, float(val))
    return best

def sum_list(vals: List[Tuple[dt.date, float]]) -> Optional[float]:
    if not vals:
        return None
    return sum(v for _, v in vals)

@lru_cache(maxsize=None)
def load_ticker_to_cik() -> Dict[str, str]:
    raw = http_get(SEC_TICKERS_URL, ua=SEC_UA)
    try: raw = gzip.decompress(raw)
    except gzip.BadGzipFile: pass
    data = json.loads(raw.decode("utf-8"))
    return {v["ticker"].upper(): str(v["cik_str"]).zfill(10) for v in data.values()}

@lru_cache(maxsize=None)
def fetch_company_facts(cik: str) -> dict:
    raw = http_get(SEC_FACTS_URL_TMPL.format(cik=cik), ua=SEC_UA)
    try: raw = gzip.decompress(raw)
    except gzip.BadGzipFile: pass
    return json.loads(raw.decode("utf-8"))

def sec_sleep():
    time.sleep(0.12)

# -----------------------------
# Alpha Vantage helpers (price + sector)
# -----------------------------
def av_symbol_variants(sym: str):
    cands = {sym}
    for a, b in [("-", "."), ("/", "."), (".", "-")]:
        if a in sym: cands.add(sym.replace(a, b))
    return list(cands)

def av_symbol_search(sym: str, api_key: str):
    base = "https://www.alphavantage.co/query"
    params = {"function": "SYMBOL_SEARCH", "keywords": sym, "apikey": api_key}
    url = base + "?" + urllib.parse.urlencode(params)
    try:
        raw = http_get(url, ua="Mozilla/5.0")
        try: raw = gzip.decompress(raw)
        except gzip.BadGzipFile: pass
        data = json.loads(raw.decode("utf-8"))
        matches = data.get("bestMatches", []) or []
        picked = []
        for m in matches:
            s = m.get("1. symbol")
            region = (m.get("4. region") or "").lower()
            currency = (m.get("8. currency") or "").upper()
            if s and ("united states" in region) and (currency in ("USD","")):
                picked.append(s)
        # unique order-preserving
        seen, out = set(), []
        for s in picked:
            if s not in seen:
                seen.add(s); out.append(s)
        return out
    except Exception:
        return []

@lru_cache(maxsize=None)
def get_price_alpha_vantage(ticker: str, api_key: str, price_date: dt.date,
                            max_retries: int = 5, throttle_sec: float = 13.0) -> Optional[float]:
    base = "https://www.alphavantage.co/query"

    def fetch_series(symbol: str):
        params = {
            "function": "TIME_SERIES_DAILY_ADJUSTED",
            "symbol": symbol,
            "outputsize": "full",
            "apikey": api_key,
        }
        url = base + "?" + urllib.parse.urlencode(params)
        for _ in range(max_retries):
            raw = http_get(url, ua="Mozilla/5.0")
            try: raw = gzip.decompress(raw)
            except gzip.BadGzipFile: pass
            try:
                data = json.loads(raw.decode("utf-8"))
            except Exception:
                data = {}

            if "Note" in data or "Information" in data:
                time.sleep(throttle_sec); continue
            if "Error Message" in data and "Invalid API call" in data["Error Message"]:
                return "INVALID", None
            series = data.get("Time Series (Daily)", {})
            if not series:
                time.sleep(throttle_sec); continue

            for dback in range(0, 8):
                d = price_date - dt.timedelta(days=dback)
                if d.isoformat() in series:
                    try: return "OK", float(series[d.isoformat()]["4. close"])
                    except Exception: return "OK", None
            return "OK", None
        return "RETRY_EXHAUSTED", None

    candidates = [ticker] + av_symbol_variants(ticker)
    candidates += [s for s in av_symbol_search(ticker, api_key) if s not in candidates]

    tried = []
    for sym in candidates:
        status, price = fetch_series(sym)
        tried.append(sym)
        if status == "OK" and price is not None:
            return price
        if status == "INVALID":
            continue
    sys.stderr.write(f"[SKIP] Unresolvable AV price for {ticker} (tried {tried})\n")
    return None

@lru_cache(maxsize=None)
def get_av_overview(ticker: str, api_key: str,
                    max_retries: int = 5, throttle_sec: float = 13.0) -> Dict[str, Optional[str]]:
    base = "https://www.alphavantage.co/query"
    def fetch(symbol: str):
        params = {"function": "OVERVIEW", "symbol": symbol, "apikey": api_key}
        url = base + "?" + urllib.parse.urlencode(params)
        for _ in range(max_retries):
            raw = http_get(url, ua="Mozilla/5.0")
            try: raw = gzip.decompress(raw)
            except gzip.BadGzipFile: pass
            try:
                data = json.loads(raw.decode("utf-8"))
            except Exception:
                data = {}
            if "Note" in data or "Information" in data:
                time.sleep(throttle_sec); continue
            if data and isinstance(data, dict) and data.get("Symbol"):
                return data
            time.sleep(throttle_sec)
        return {}

    # Try variants + search as with price
    candidates = [ticker] + av_symbol_variants(ticker)
    candidates += [s for s in av_symbol_search(ticker, api_key) if s not in candidates]
    for sym in candidates:
        data = fetch(sym)
        if data:
            return {
                "Sector": data.get("Sector"),
                "Industry": data.get("Industry"),
                "SharesOutstanding": data.get("SharesOutstanding"),
            }
    return {"Sector": None, "Industry": None, "SharesOutstanding": None}

# -----------------------------
# Feature engineering
# -----------------------------
def build_features_from_facts(facts: dict) -> Dict[str, Optional[float]]:
    rev_q = pick_last4_quarterly(facts, TAGSETS["revenue"])
    ni_q  = pick_last4_quarterly(facts, TAGSETS["net_income"])
    op_q  = pick_last4_quarterly(facts, TAGSETS["operating_income"])
    da_q  = pick_last4_quarterly(facts, TAGSETS["dep_amort"])
    gp_q  = pick_last4_quarterly(facts, TAGSETS["gross_profit"])
    int_exp_q = pick_last4_quarterly(facts, TAGSETS["interest_expense"])
    capex_q   = pick_last4_quarterly(facts, TAGSETS["capex"])
    ocf_q     = pick_last4_quarterly(facts, TAGSETS["operating_cf"])

    assets_latest = latest_value(facts, TAGSETS["assets"])
    equity_latest = latest_value(facts, TAGSETS["equity"])
    debt_lt_latest = latest_value(facts, TAGSETS["debt_longterm"])
    debt_cur_latest = latest_value(facts, TAGSETS["debt_current"])
    stb_latest = latest_value(facts, TAGSETS["short_term_borrow"])
    cash_latest = latest_value(facts, TAGSETS["cash_and_equiv"])
    ca_latest = latest_value(facts, TAGSETS["assets_current"])
    cl_latest = latest_value(facts, TAGSETS["liabilities_current"])
    shares_diluted_latest = latest_value(facts, TAGSETS["shares_diluted"])

    ttm_rev = sum_list(rev_q)
    ttm_ni  = sum_list(ni_q)
    ttm_op  = sum_list(op_q)
    ttm_da  = sum_list(da_q)
    ttm_gp  = sum_list(gp_q)
    ttm_int = sum_list(int_exp_q)
    ttm_capex = sum_list(capex_q)
    ttm_ocf = sum_list(ocf_q)

    ebitda_ttm = (ttm_op or 0.0) + (ttm_da or 0.0) if (ttm_op is not None or ttm_da is not None) else None
    current_ratio = (ca_latest[1] / cl_latest[1]) if (ca_latest and cl_latest and cl_latest[1] not in (None, 0)) else None

    equity_val = equity_latest[1] if equity_latest else None
    assets_val = assets_latest[1] if assets_latest else None
    debt_parts = [
        debt_lt_latest[1] if debt_lt_latest else None,
        debt_cur_latest[1] if debt_cur_latest else None,
        stb_latest[1] if stb_latest else None
    ]
    debt_total = None if all(v is None for v in debt_parts) else sum(0.0 if v is None else v for v in debt_parts)

    roe = (ttm_ni / equity_val) if (ttm_ni is not None and equity_val not in (None, 0)) else None
    roa = (ttm_ni / assets_val) if (ttm_ni is not None and assets_val not in (None, 0)) else None
    interest_coverage = (ttm_op / ttm_int) if (ttm_op is not None and ttm_int not in (None, 0)) else None

    return {
        "revenue_ttm": ttm_rev,
        "net_income_ttm": ttm_ni,
        "ebitda_ttm": ebitda_ttm,
        "gross_profit_ttm": ttm_gp,
        "operating_income_ttm": ttm_op,
        "interest_expense_ttm": ttm_int,
        "capex_ttm": ttm_capex,
        "operating_cf_ttm": ttm_ocf,
        "gross_margin": (ttm_gp / ttm_rev) if (ttm_gp is not None and ttm_rev not in (None, 0)) else None,
        "operating_margin": (ttm_op / ttm_rev) if (ttm_op is not None and ttm_rev not in (None, 0)) else None,
        "net_margin": (ttm_ni / ttm_rev) if (ttm_ni is not None and ttm_rev not in (None, 0)) else None,
        "roe": roe, "roa": roa,
        "equity_latest": equity_val,
        "assets_latest": assets_val,
        "debt_total_latest": debt_total,
        "cash_and_equivalents": (cash_latest[1] if cash_latest else None),
        "current_ratio": current_ratio,
        "interest_coverage": interest_coverage,
        "shares_diluted_latest": (shares_diluted_latest[1] if shares_diluted_latest else None),
    }

# -----------------------------
# Scoring helpers (sector-aware)
# -----------------------------
def _winsorize(s: pd.Series, low=0.01, high=0.99) -> pd.Series:
    lo, hi = s.quantile(low), s.quantile(high)
    return s.clip(lo, hi)

def _decile(series: pd.Series, invert: bool = False) -> pd.Series:
    r = series.rank(pct=True, na_option="keep")
    s = 1 + 9 * r  # 1..10
    return (10 - s) if invert else s

def _sector_decile(df: pd.DataFrame, col: str, sector_col: str, invert: bool = False) -> pd.Series:
    def _rank(g):
        r = g.rank(pct=True, na_option="keep")
        s = 1 + 9 * r
        return (10 - s) if invert else s
    return df.groupby(sector_col, dropna=False)[col].transform(_rank)

def compute_sector_aware_score(df: pd.DataFrame, sector_col="sector",
                               quality_max=1.0, risk_max=1.5, redflag_add=0.5,
                               w_abs=0.50, w_rel=0.30, sector_relief_max=0.8) -> pd.DataFrame:
    out = df.copy()

    # Derived basics
    out["mkt_cap"] = out["price_now"] * out["shares_diluted_latest"]
    out.loc[(out["shares_diluted_latest"]<=0) | out["shares_diluted_latest"].isna(), "mkt_cap"] = np.nan
    out["ev"] = out["mkt_cap"] + out["debt_total_latest"].fillna(0) - out["cash_and_equivalents"].fillna(0)

    eps = out["net_income_ttm"] / out["shares_diluted_latest"]
    out["pe_ttm"] = out["price_now"] / eps
    out.loc[(eps <= 0) | (out["shares_diluted_latest"]<=0), "pe_ttm"] = np.nan

    out["ev_ebitda"] = out["ev"] / out["ebitda_ttm"]
    out.loc[(out["ebitda_ttm"]<=0) | out["ev"].isna(), "ev_ebitda"] = np.nan

    out["pb"] = out["mkt_cap"] / out["equity_latest"]
    out["ps"] = out["mkt_cap"] / out["revenue_ttm"]

    if {"operating_cf_ttm","capex_ttm"}.issubset(out.columns):
        out["fcf_ttm"] = out["operating_cf_ttm"] - out["capex_ttm"]
        out["fcf_yield"] = out["fcf_ttm"] / out["mkt_cap"]
    else:
        out["fcf_yield"] = np.nan

    out["roe_calc"] = out["net_income_ttm"] / out["equity_latest"]
    out["ebitda_margin"] = out["ebitda_ttm"] / out["revenue_ttm"]
    out["net_debt_ebitda"] = (out["debt_total_latest"] - out["cash_and_equivalents"]) / out["ebitda_ttm"]
    out.loc[out["ebitda_ttm"]<=0, "net_debt_ebitda"] = np.nan

    # Winsorize
    for col in ["pe_ttm","ev_ebitda","pb","ps","fcf_yield","roe_calc","ebitda_margin","net_debt_ebitda","interest_coverage"]:
        if col in out.columns:
            out[col] = _winsorize(out[col])

    # Absolute value deciles (higher => more overvalued)
    V_abs_parts = [
        _decile(out["pe_ttm"]),
        _decile(out["ev_ebitda"]),
        _decile(out["pb"]),
        _decile(out["ps"]),
        _decile(out["fcf_yield"], invert=True),
    ]
    out["V_abs"] = np.nanmean(np.vstack([p.to_numpy() for p in V_abs_parts]), axis=0)

    # Sector-relative value deciles
    out["V_rel"] = np.nanmean(np.vstack([
        _sector_decile(out, "pe_ttm", sector_col),
        _sector_decile(out, "ev_ebitda", sector_col),
        _sector_decile(out, "pb", sector_col),
        _sector_decile(out, "ps", sector_col),
        _sector_decile(out, "fcf_yield", sector_col, invert=True),
    ]), axis=0)

    # Quality bonus (subtract points)
    Q_parts = [
        _decile(out["roe_calc"], invert=True),
        _decile(out["ebitda_margin"], invert=True),
    ]
    out["quality_bonus_pts"] = (np.nanmean(np.vstack([p.to_numpy() for p in Q_parts]), axis=0) / 10.0) * quality_max

    # Risk penalty (add points)
    R_parts = [
        _decile(out["net_debt_ebitda"]),
        _decile(1.0 / out["interest_coverage"]),
    ]
    out["risk_penalty_pts"] = (np.nanmean(np.vstack([p.to_numpy() for p in R_parts]), axis=0) / 10.0) * risk_max

    # Red flags
    redflag = (
        (eps <= 0) |
        (out["ebitda_ttm"] <= 0) |
        (out["interest_coverage"] < 1.5) |
        (out.get("current_ratio", np.inf) < 1.0)
    )
    out["redflag_pts"] = np.where(redflag, redflag_add, 0.0)

    # Sector relief: if a sector is broadly expensive, reduce penalty a bit
    sector_median = out.groupby(sector_col, dropna=False)["V_abs"].transform("median")
    univ_median = out["V_abs"].median()
    out["sector_relief_pts"] = np.clip(sector_median - univ_median, 0, 2.0) * (sector_relief_max / 2.0)

    # Final 1..10 score
    score = (w_abs * out["V_abs"]) + (w_rel * out["V_rel"]) \
            + out["risk_penalty_pts"] + out["redflag_pts"] \
            - out["quality_bonus_pts"] - out["sector_relief_pts"]

    out["undervaluation_score"] = score.clip(1, 10).round(1)

    # Select/export columns
    cols = [
        "ticker","sector","undervaluation_score",
        "V_abs","V_rel","quality_bonus_pts","risk_penalty_pts","redflag_pts","sector_relief_pts",
        "pe_ttm","ev_ebitda","pb","ps","fcf_yield",
        "roe_calc","ebitda_margin","net_debt_ebitda","interest_coverage",
        "mkt_cap","ev",
        "price_now","shares_diluted_latest","revenue_ttm","net_income_ttm","ebitda_ttm",
        "equity_latest","debt_total_latest","cash_and_equivalents","current_ratio"
    ]
    cols = [c for c in cols if c in out.columns]
    return out[cols]

# -----------------------------
# Dataset build + scoring
# -----------------------------
def build_dataset_and_score(universe: List[str], now_date: dt.date,
                            price_api_key: str, sectors_csv: Optional[str],
                            price_max_retries: int, price_throttle_sec: float,
                            outdir: str) -> pd.DataFrame:
    os.makedirs(outdir, exist_ok=True)
    t2c = load_ticker_to_cik()
    rows = []

    # Preload sectors CSV if provided
    sectors_map = {}
    if sectors_csv:
        s = pd.read_csv(sectors_csv)
        if {"ticker","sector"}.issubset(s.columns):
            sectors_map = {str(r["ticker"]).upper(): r["sector"] for _, r in s.iterrows()}

    for i, t in enumerate(universe):
        tU = t.upper().strip()
        cik = t2c.get(tU)
        if not cik:
            print(f"[SKIP] No CIK for {tU}", file=sys.stderr)
            continue
        try:
            # SEC facts
            facts = fetch_company_facts(cik)
            feats = build_features_from_facts(facts)

            # Price (near now_date)
            price_now = get_price_alpha_vantage(tU, price_api_key, now_date,
                                                max_retries=price_max_retries,
                                                throttle_sec=price_throttle_sec)
            time.sleep(price_throttle_sec)

            if price_now is None:
                print(f"[SKIP] Missing price for {tU}", file=sys.stderr)
                continue

            # Sector: CSV override > AV Overview > Unknown
            sector = sectors_map.get(tU)
            if sector is None:
                ov = get_av_overview(tU, price_api_key,
                                     max_retries=price_max_retries,
                                     throttle_sec=price_throttle_sec)
                time.sleep(price_throttle_sec)
                sector = ov.get("Sector") or "Unknown"

                # Optional: If SEC shares missing, use AV SharesOutstanding to allow mkt cap calc
                if feats.get("shares_diluted_latest") in (None, 0) and ov.get("SharesOutstanding"):
                    try:
                        feats["shares_diluted_latest"] = float(ov["SharesOutstanding"])
                    except Exception:
                        pass

            row = {"ticker": tU, "price_now": price_now, "sector": sector}
            row.update(feats)
            rows.append(row)
            sec_sleep()

            if (i + 1) % 10 == 0:
                print(f"Processed {i+1}/{len(universe)} tickers...", flush=True)

        except Exception as e:
            print(f"[WARN] {tU}: {e}", file=sys.stderr)
            continue

    base = pd.DataFrame(rows)
    base_path = os.path.join(outdir, "dataset_quarterly_raw.csv")
    base.to_csv(base_path, index=False)

    # Score (sector-aware)
    scored = compute_sector_aware_score(base, sector_col="sector")
    scored.sort_values(["sector","undervaluation_score","ticker"], inplace=True)

    # Write outputs
    scored_path = os.path.join(outdir, "undervaluation_scored.csv")
    scored.to_csv(scored_path, index=False)

    # Also write a Top 10 per sector convenience file
    top_rows = []
    for sec, g in scored.groupby("sector", dropna=False):
        top_rows.append(g.nsmallest(10, "undervaluation_score"))
    top_by_sector = pd.concat(top_rows) if top_rows else scored.head(0)
    top_path = os.path.join(outdir, "top10_by_sector.csv")
    top_by_sector.to_csv(top_path, index=False)

    print("Saved:", base_path)
    print("Saved:", scored_path)
    print("Saved:", top_path)
    return scored

# -----------------------------
# CLI
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--universe_csv", required=True, help="CSV with a 'ticker' column (case-insensitive).")
    ap.add_argument("--now_date", type=str, default=dt.date.today().isoformat(), help="Valuation as of date (YYYY-MM-DD).")
    ap.add_argument("--price_vendor", type=str, required=True, choices=["alpha_vantage"], help="Price data vendor.")
    ap.add_argument("--price_api_key", type=str, required=True, help="Alpha Vantage API key.")
    ap.add_argument("--sectors_csv", type=str, default=None, help="Optional CSV with columns: ticker,sector (overrides AV sector).")
    ap.add_argument("--outdir", type=str, default="./val_output", help="Output directory.")
    ap.add_argument("--av_throttle_sec", type=float, default=13.0, help="Seconds between AV calls.")
    ap.add_argument("--av_max_retries", type=int, default=5, help="Max retries for AV calls.")
    args = ap.parse_args()

    # Load universe
    with open(args.universe_csv, "r", newline="") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        if header and any(h.lower() == "ticker" for h in header):
            idx = next(i for i, h in enumerate(header) if h.lower() == "ticker")
            tickers = [row[idx].strip().upper() for row in reader if row]
        else:
            tickers = [row[0].strip().upper() for row in reader if row]

    now_date = dt.date.fromisoformat(args.now_date)

    # Build + score
    build_dataset_and_score(
        universe=tickers,
        now_date=now_date,
        price_api_key=args.price_api_key,
        sectors_csv=args.sectors_csv,
        price_max_retries=args.av_max_retries,
        price_throttle_sec=args.av_throttle_sec,
        outdir=args.outdir
    )

if __name__ == "__main__":
    main()
