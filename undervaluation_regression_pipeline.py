#!/usr/bin/env python3
# Undervaluation Regression Pipeline (SEC + Alpha Vantage)
# Author: ChatGPT (for ANGAD ARORA)
# License: MIT
#
# Adds P/E (TTM) feature and supports multi-year growth horizon (default 3y).
# If --anchor_date is omitted, we compute anchor_date = now_date minus horizon_years.

import argparse
import csv
import datetime as dt
import json
import math
import os
import sys
import time
import urllib.parse
import urllib.request
import gzip
from functools import lru_cache
from typing import Dict, List, Optional, Tuple

# ---- Robust imports (headless matplotlib + clear errors) ----
try:
    import numpy as np
    import pandas as pd
    import matplotlib
    matplotlib.use("Agg")  # headless backend
    import matplotlib.pyplot as plt
    from sklearn.model_selection import train_test_split
except ImportError as e:
    pkg = getattr(e, "name", str(e))
    print(f"Missing package: {pkg}")
    print("Install with: python3 -m pip install numpy pandas matplotlib scikit-learn")
    sys.exit(1)
except Exception as e:
    print("Import failed:", repr(e))
    print("Try: python3 -m pip install --upgrade pip setuptools wheel")
    print("Then: python3 -m pip install --upgrade numpy pandas matplotlib scikit-learn")
    sys.exit(1)

# -----------------------------
# Config & Constants
# -----------------------------

# 👉 IMPORTANT: use a real UA + contact email per SEC guidance
SEC_UA = "UndervaluationPipeline/1.0 (erangadarora@gmail.com)"
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
    # for EPS/PE:
    "shares_diluted": ["WeightedAverageNumberOfDilutedSharesOutstanding"],
}

# -----------------------------
# Helpers
# -----------------------------

def http_get(url: str, ua: str, timeout: int = 30) -> bytes:
    """HTTP GET with UA; caller handles JSON+gzip."""
    req = urllib.request.Request(url, headers={"User-Agent": ua, "Accept-Encoding": "gzip, deflate"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()

def to_date(s: str) -> Optional[dt.date]:
    try:
        return dt.date.fromisoformat(s[:10])
    except Exception:
        return None

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
    # Correct nesting: facts -> us-gaap
    gaap = facts.get("facts", {}).get(US_GAAP, {})
    for t in taglist:
        node = gaap.get(t, {})
        for _, arr in node.get("units", {}).items():
            q = last_quarters(arr, within_days=600, want=4)
            if q:
                return q
    return []

def latest_value(facts: dict, taglist: List[str]):
    # Correct nesting: facts -> us-gaap
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

def ridge_closed_form(X: np.ndarray, y: np.ndarray, alpha: float = 1.0):
    ones = np.ones((X.shape[0], 1))
    Xb = np.hstack([ones, X])
    I = np.eye(Xb.shape[1]); I[0, 0] = 0.0
    XtX = Xb.T @ Xb; Xty = Xb.T @ y
    beta_all = np.linalg.pinv(XtX + alpha * I) @ Xty
    return beta_all[1:], beta_all[0]

def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else float('nan')

def permutation_importance(X: np.ndarray, y: np.ndarray, beta: np.ndarray, intercept: float,
                           n_repeats: int = 20, random_state: int = 42):
    rng = np.random.default_rng(random_state)
    base_r2 = r2_score(y, X @ beta + intercept)
    imps = np.zeros(X.shape[1])
    for j in range(X.shape[1]):
        r2_shuffled = []
        for _ in range(n_repeats):
            Xp = X.copy()
            rng.shuffle(Xp[:, j])
            r2p = r2_score(y, Xp @ beta + intercept)
            r2_shuffled.append(r2p)
        imps[j] = base_r2 - np.mean(r2_shuffled)
    return imps

def years_ago(d: dt.date, years: int) -> dt.date:
    """Return a date ~N years before d; handle Feb 29 gracefully."""
    try:
        return d.replace(year=d.year - years)
    except ValueError:
        # e.g., Feb 29 -> Feb 28
        return d.replace(month=2, day=28, year=d.year - years)

# -----------------------------
# Data Acquisition (SEC)
# -----------------------------

@lru_cache(maxsize=None)
def load_ticker_to_cik() -> Dict[str, str]:
    raw = http_get(SEC_TICKERS_URL, ua=SEC_UA)
    try:
        raw = gzip.decompress(raw)
    except gzip.BadGzipFile:
        pass
    data = json.loads(raw.decode("utf-8"))
    return {v["ticker"].upper(): str(v["cik_str"]).zfill(10) for v in data.values()}

@lru_cache(maxsize=None)
def fetch_company_facts(cik: str) -> dict:
    raw = http_get(SEC_FACTS_URL_TMPL.format(cik=cik), ua=SEC_UA)
    try:
        raw = gzip.decompress(raw)
    except gzip.BadGzipFile:
        pass
    return json.loads(raw.decode("utf-8"))

def sec_sleep():
    time.sleep(0.12)  # be nice to SEC API

# -----------------------------
# Price Vendor (Alpha Vantage)
# -----------------------------

@lru_cache(maxsize=None)
def get_price_alpha_vantage(ticker: str, api_key: str, price_date: dt.date,
                            max_retries: int = 5, throttle_sec: float = 13.0) -> Optional[float]:
    """
    Premium endpoint: TIME_SERIES_DAILY_ADJUSTED.
    Retries/backoff for rate messages; finds close within 7-day lookback.
    """
    base = "https://www.alphavantage.co/query"
    params = {
        "function": "TIME_SERIES_DAILY_ADJUSTED",
        "symbol": ticker,
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

        if "Error Message" in data:
            raise RuntimeError(f"Alpha Vantage error for {ticker}: {data['Error Message']}")
        if "Note" in data or "Information" in data:
            time.sleep(throttle_sec); continue

        series = data.get("Time Series (Daily)", {})
        if not series:
            time.sleep(throttle_sec); continue

        for dback in range(0, 8):  # weekends/holidays
            d = price_date - dt.timedelta(days=dback)
            key = d.isoformat()
            if key in series:
                try:
                    return float(series[key]["4. close"])
                except Exception:
                    return None

        return None  # series ok but not within 7 days

    return None  # exhausted retries

# -----------------------------
# Feature Engineering & Dataset
# -----------------------------

def build_features_from_facts(facts: dict) -> Dict[str, Optional[float]]:
    rev_q = pick_last4_quarterly(facts, TAGSETS["revenue"])
    ni_q  = pick_last4_quarterly(facts, TAGSETS["net_income"])
    op_q  = pick_last4_quarterly(facts, TAGSETS["operating_income"])
    da_q  = pick_last4_quarterly(facts, TAGSETS["dep_amort"])
    gp_q  = pick_last4_quarterly(facts, TAGSETS["gross_profit"])

    assets_latest = latest_value(facts, TAGSETS["assets"])
    equity_latest = latest_value(facts, TAGSETS["equity"])
    debt_lt_latest = latest_value(facts, TAGSETS["debt_longterm"])
    debt_cur_latest = latest_value(facts, TAGSETS["debt_current"])
    stb_latest = latest_value(facts, TAGSETS["short_term_borrow"])
    cash_latest = latest_value(facts, TAGSETS["cash_and_equiv"])
    ca_latest = latest_value(facts, TAGSETS["assets_current"])
    cl_latest = latest_value(facts, TAGSETS["liabilities_current"])
    int_exp_q = pick_last4_quarterly(facts, TAGSETS["interest_expense"])
    capex_q   = pick_last4_quarterly(facts, TAGSETS["capex"])
    shares_diluted_latest = latest_value(facts, TAGSETS["shares_diluted"])

    ttm_rev = sum_list(rev_q)
    ttm_ni  = sum_list(ni_q)
    ttm_op  = sum_list(op_q)
    ttm_da  = sum_list(da_q)
    ttm_gp  = sum_list(gp_q)
    ttm_int = sum_list(int_exp_q)
    _ttm_capex = sum_list(capex_q)  # not used below

    # EBITDA TTM
    ebitda_ttm = (ttm_op or 0.0) + (ttm_da or 0.0) if (ttm_op is not None or ttm_da is not None) else None

    # Margins
    gross_margin = (ttm_gp / ttm_rev) if (ttm_gp is not None and ttm_rev not in (None, 0)) else None
    op_margin    = (ttm_op / ttm_rev) if (ttm_op is not None and ttm_rev not in (None, 0)) else None
    net_margin   = (ttm_ni / ttm_rev) if (ttm_ni is not None and ttm_rev not in (None, 0)) else None

    equity_val = equity_latest[1] if equity_latest else None
    assets_val = assets_latest[1] if assets_latest else None

    # Debt total (None if all parts missing)
    debt_parts = [
        debt_lt_latest[1] if debt_lt_latest else None,
        debt_cur_latest[1] if debt_cur_latest else None,
        stb_latest[1] if stb_latest else None
    ]
    debt_total = None if all(v is None for v in debt_parts) else sum(0.0 if v is None else v for v in debt_parts)

    roe = (ttm_ni / equity_val) if (ttm_ni is not None and equity_val not in (None, 0)) else None
    roa = (ttm_ni / assets_val) if (ttm_ni is not None and assets_val not in (None, 0)) else None
    interest_coverage = (ttm_op / ttm_int) if (ttm_op is not None and ttm_int not in (None, 0)) else None
    current_ratio = (ca_latest[1] / cl_latest[1]) if (ca_latest and cl_latest and cl_latest[1] not in (None, 0)) else None

    return {
        "revenue_ttm": ttm_rev,
        "net_income_ttm": ttm_ni,
        "ebitda_ttm": ebitda_ttm,
        "gross_margin": gross_margin,
        "operating_margin": op_margin,
        "net_margin": net_margin,
        "roe": roe,
        "roa": roa,
        "debt_to_equity": (debt_total / equity_val) if (debt_total is not None and equity_val not in (None, 0)) else None,
        "interest_coverage": interest_coverage,
        "current_ratio": current_ratio,
        "cash_and_equivalents": (cash_latest[1] if cash_latest else None),
        "total_debt_latest": debt_total,
        # helpers exposed for PE calc (not used as model features directly):
        "shares_diluted_latest": (shares_diluted_latest[1] if shares_diluted_latest else None),
        "equity_latest": equity_val,
        "assets_latest": assets_val,
    }

def build_dataset(universe: List[str], anchor_date: dt.date, now_date: dt.date,
                  price_vendor: str, price_api_key: str,
                  price_max_retries: int = 5, price_throttle_sec: float = 13.0,
                  min_non_null_features: int = 3, horizon_years: int = 3) -> pd.DataFrame:
    t2c = load_ticker_to_cik()
    rows = []
    target_col_name = f"growth_{horizon_years}y"

    for i, t in enumerate(universe):
        tU = t.upper().strip()
        cik = t2c.get(tU)
        if not cik:
            print(f"[SKIP] No CIK for {tU}", file=sys.stderr)
            continue
        try:
            facts = fetch_company_facts(cik)
            feats = build_features_from_facts(facts)

            # Prices (with pacing to respect quota)
            price_then = get_price_alpha_vantage(tU, price_api_key, anchor_date,
                                                 max_retries=price_max_retries,
                                                 throttle_sec=price_throttle_sec)
            time.sleep(price_throttle_sec)
            price_now  = get_price_alpha_vantage(tU, price_api_key, now_date,
                                                 max_retries=price_max_retries,
                                                 throttle_sec=price_throttle_sec)
            time.sleep(price_throttle_sec)

            if price_then is None or price_now is None or price_then == 0:
                print(f"[SKIP] Missing/invalid prices for {tU}", file=sys.stderr)
                continue

            growth = (price_now - price_then) / price_then

            # Compute P/E (TTM) only if EPS>0
            pe_ttm = None
            ttm_ni = feats.get("net_income_ttm")
            shares_diluted = feats.get("shares_diluted_latest")
            if ttm_ni is not None and shares_diluted not in (None, 0):
                eps_ttm = ttm_ni / shares_diluted
                if eps_ttm and eps_ttm > 0:
                    pe_ttm = price_now / eps_ttm

            # Require a minimum number of usable SEC features (exclude helper-only fields)
            core_feats_keys = [
                "revenue_ttm","net_income_ttm","ebitda_ttm",
                "gross_margin","operating_margin","net_margin",
                "roe","roa","debt_to_equity","interest_coverage",
                "current_ratio","cash_and_equivalents","total_debt_latest"
            ]
            non_null_feats = sum(feats[k] is not None for k in core_feats_keys)
            if non_null_feats < min_non_null_features:
                print(f"[SKIP] Too few SEC features ({non_null_feats}) for {tU}", file=sys.stderr)
                continue

            row = {
                "ticker": tU,
                "price_then": price_then,
                "price_now": price_now,
                target_col_name: growth,
                "pe_ttm": pe_ttm,
            }
            for k in core_feats_keys:
                row[k] = feats.get(k)

            rows.append(row)
            sec_sleep()  # politeness to SEC

            if (i + 1) % 10 == 0:
                print(f"Processed {i+1}/{len(universe)} tickers...", flush=True)

        except Exception as e:
            print(f"[WARN] {tU}: {e}", file=sys.stderr)
            continue

    return pd.DataFrame(rows)

# -----------------------------
# Modeling
# -----------------------------

def standardize_frame(df: pd.DataFrame, feature_cols: List[str]):
    X = df[feature_cols].astype(float)
    keep, stats = [], {}
    for c in feature_cols:
        col = X[c]
        if col.isna().all():
            continue
        std, mean = col.std(ddof=0), col.mean()
        if std == 0.0 or math.isclose(std, 0.0):
            continue
        X.loc[:, c] = (col - mean) / std
        keep.append(c); stats[c] = (mean, std)
    return X[keep].to_numpy(), keep, stats

def train_ridge(df: pd.DataFrame, feature_cols: List[str], target_col: str, alpha: float, test_size: float, seed: int):
    data = df.dropna(subset=[target_col]).copy()
    data = data.dropna(axis=0, how='all', subset=feature_cols)
    data = data.fillna(data.median(numeric_only=True))

    if len(data) < 20:
        raise ValueError("Not enough complete rows to train. Consider a larger universe or different features.")

    train, test = train_test_split(data, test_size=test_size, random_state=seed)

    X_train, kept_cols, train_stats = standardize_frame(train, feature_cols)
    y_train = train[target_col].to_numpy(dtype=float)

    X_test = test[kept_cols].copy()
    for c in kept_cols:
        X_test.loc[:, c] = (test[c] - train_stats[c][0]) / train_stats[c][1]
    X_test = X_test.to_numpy()
    y_test = test[target_col].to_numpy(dtype=float)

    beta, intercept = ridge_closed_form(X_train, y_train, alpha=alpha)
    r2_train = r2_score(y_train, X_train @ beta + intercept)
    r2_test  = r2_score(y_test,  X_test  @ beta + intercept)

    coef_importance = (
        pd.DataFrame({"feature": kept_cols, "coef": beta, "abs_coef": np.abs(beta)})
        .sort_values("abs_coef", ascending=False)
    )
    pimps = permutation_importance(X_test, y_test, beta, intercept, n_repeats=50, random_state=seed)
    perm_importance = (
        pd.DataFrame({"feature": kept_cols, "importance_drop_r2": pimps})
        .sort_values("importance_drop_r2", ascending=False)
    )

    report = {"n_train": len(train), "n_test": len(test), "alpha": alpha, "r2_train": r2_train, "r2_test": r2_test}
    return report, coef_importance, perm_importance

# -----------------------------
# CLI
# -----------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--universe_csv", type=str, required=True, help="CSV with a 'ticker' column (case-insensitive).")
    ap.add_argument("--anchor_date", type=str, default=None, help="Historical date (YYYY-MM-DD). If omitted, computed from horizon_years.")
    ap.add_argument("--now_date", type=str, default=dt.date.today().isoformat(), help="Comparison date (YYYY-MM-DD).")
    ap.add_argument("--horizon_years", type=int, default=3, help="If anchor_date not given, use now_date minus this many years.")
    ap.add_argument("--price_vendor", type=str, required=True, choices=["alpha_vantage"], help="Price data vendor.")
    ap.add_argument("--price_api_key", type=str, required=True, help="Alpha Vantage API key (premium for adjusted).")
    ap.add_argument("--alpha", type=float, default=1.0, help="Ridge regularization strength.")
    ap.add_argument("--test_size", type=float, default=0.25, help="Test set fraction.")
    ap.add_argument("--outdir", type=str, default="./output", help="Output directory.")
    ap.add_argument("--av_throttle_sec", type=float, default=13.0, help="Seconds to sleep between AV calls.")
    ap.add_argument("--av_max_retries", type=int, default=5, help="Max retries for AV calls when throttled.")
    ap.add_argument("--min_feats", type=int, default=3, help="Minimum non-null SEC features required to keep a row.")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    # Load universe
    with open(args.universe_csv, "r", newline="") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        if header and any(h.lower() == "ticker" for h in header):
            ticker_idx = next(i for i, h in enumerate(header) if h.lower() == "ticker")
            tickers = [row[ticker_idx].strip().upper() for row in reader if row]
        else:
            tickers = [row[0].strip().upper() for row in reader if row]

    # Resolve dates
    now_date = dt.date.fromisoformat(args.now_date)
    if args.anchor_date:
        anchor_date = dt.date.fromisoformat(args.anchor_date)
        horizon_years = round((now_date.year - anchor_date.year) + ((now_date - anchor_date).days / 365.25))
    else:
        horizon_years = args.horizon_years
        anchor_date = years_ago(now_date, horizon_years)

    # Build dataset
    df = build_dataset(
        universe=tickers,
        anchor_date=anchor_date,
        now_date=now_date,
        price_vendor=args.price_vendor,
        price_api_key=args.price_api_key,
        price_max_retries=args.av_max_retries,
        price_throttle_sec=args.av_throttle_sec,
        min_non_null_features=args.min_feats,
        horizon_years=horizon_years
    )

    # Save raw dataset
    raw_path = os.path.join(args.outdir, "dataset_raw.csv")
    df.to_csv(raw_path, index=False)

    # Features for modeling (includes P/E)
    feature_cols = [
        "revenue_ttm","net_income_ttm","ebitda_ttm",
        "gross_margin","operating_margin","net_margin",
        "roe","roa","debt_to_equity","interest_coverage",
        "current_ratio","cash_and_equivalents","total_debt_latest",
        "pe_ttm"
    ]

    # Determine target column name from horizon
    target_col = f"growth_{horizon_years}y"

    # Save a filtered/survivor dataset for transparency
    survivors = df.dropna(subset=[target_col]).dropna(axis=0, how="all", subset=feature_cols)
    survivors_path = os.path.join(args.outdir, "dataset_filtered.csv")
    survivors.to_csv(survivors_path, index=False)

    if len(survivors) < 20:
        print(f"[INFO] Only {len(survivors)} usable rows after filtering; training may fail.\n"
              f"See {survivors_path} for details.", file=sys.stderr)

    # Train ridge
    report, coef_imp, perm_imp = train_ridge(
        df, feature_cols, target_col=target_col, alpha=args.alpha, test_size=args.test_size, seed=42
    )

    # Write report
    with open(os.path.join(args.outdir, "model_report.txt"), "w") as f:
        for k, v in report.items():
            f.write(f"{k}: {v}\n")

    # Write importances
    coef_file = os.path.join(args.outdir, "feature_importance_coefficients.csv")
    perm_file = os.path.join(args.outdir, "feature_importance_permutation.csv")
    coef_imp.to_csv(coef_file, index=False)
    perm_imp.to_csv(perm_file, index=False)

    # Plot importances (only if non-empty)
    if not coef_imp.empty:
        plt.figure()
        plt.barh(coef_imp["feature"], coef_imp["abs_coef"])
        plt.gca().invert_yaxis()
        plt.title("Feature Importance (|coef|)")
        plt.tight_layout()
        plt.savefig(os.path.join(args.outdir, "feature_importance.png"))
        plt.close()

    print("Done.")
    print("Outputs in:", args.outdir)
    print(f"Horizon years: {horizon_years}  |  Anchor: {anchor_date}  |  Now: {now_date}")
    print(f"R^2 Test: {report['r2_test']:.3f}")

if __name__ == "__main__":
    main()
