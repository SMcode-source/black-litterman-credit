"""Utility functions for bond parsing, YTM calculation, and column resolution."""
import math
from datetime import datetime, date

import pandas as pd


# ── Default PD by rating (annualised, approximate Moody's) ──

RATING_PD_MAP = {
    "AAA": 0.0001, "AA+": 0.0002, "AA": 0.0003, "AA-": 0.0004,
    "A+": 0.0006, "A": 0.0008, "A-": 0.0012,
    "BBB+": 0.0020, "BBB": 0.0035, "BBB-": 0.0055,
    "BB+": 0.0080, "BB": 0.0120, "BB-": 0.0180,
    "B+": 0.0300, "B": 0.0500, "B-": 0.0800,
}

# ── Rating-based OAS estimates (bp) ──

RATING_SPREAD_MAP = {
    "AAA": 35, "AA+": 45, "AA": 50, "AA-": 55,
    "A+": 65, "A": 75, "A-": 85,
    "BBB+": 105, "BBB": 130, "BBB-": 160,
    "BB+": 220, "BB": 300, "BB-": 400,
}

# ── Flexible column name mapping ──

COLUMN_ALIASES = {
    "id": ["id", "ticker", "symbol", "bond_id", "isin", "cusip", "identifier"],
    "issuer": ["issuer", "issuer_name", "name", "company", "entity"],
    "sector": ["sector", "industry", "sector_name", "industry_group"],
    "rating": ["rating", "credit_rating", "sp_rating", "moody_rating", "fitch_rating"],
    "oas": ["oas", "option_adjusted_spread", "spread", "z_spread", "oas_bp", "spread_bp"],
    "spread_duration": ["spread_duration", "spread_dur", "spd_dur", "sd", "sprd_dur", "duration"],
    "pd_annual": ["pd_annual", "pd", "default_prob", "probability_of_default", "annual_pd"],
    "lgd": ["lgd", "loss_given_default", "severity"],
    "mkt_value": ["mkt_value", "market_value", "mv", "notional", "par_amount", "amount", "weight"],
    "maturity_bucket": ["maturity_bucket", "mat_bucket", "bucket", "tenor_bucket"],
    "coupon_rate": ["coupon_rate", "coupon", "cpn", "coupon_pct"],
    "maturity_date": ["maturity_date", "maturity", "mat_date", "matdate"],
    "face_value": ["face_value", "par_value", "par", "face", "fv"],
    "market_price": ["market_price", "price", "clean_price", "dirty_price", "px"],
    "ytm": ["ytm", "yield", "yield_to_maturity", "yld", "yield_pct"],
}


def compute_ytm(face_value: float, market_price: float,
                coupon_rate: float, years_to_maturity: float,
                freq: int = 2) -> float:
    """Newton-Raphson YTM solver. Returns annual yield in percent."""
    if years_to_maturity <= 0 or market_price <= 0:
        return 0.0
    coupon = face_value * (coupon_rate / 100.0) / freq
    n_periods = max(int(round(years_to_maturity * freq)), 1)
    ytm_guess = (coupon * freq) / market_price

    for _ in range(200):
        r = max(ytm_guess / freq, 1e-12)
        disc = 1 + r
        pv_coupons = coupon * (1 - disc ** (-n_periods)) / r
        price = pv_coupons + face_value * disc ** (-n_periods)
        # Numerical derivative
        r2 = r + 0.0001
        disc2 = 1 + r2
        price2 = coupon * (1 - disc2 ** (-n_periods)) / r2 + face_value * disc2 ** (-n_periods)
        deriv = (price2 - price) / 0.0001
        if abs(deriv) < 1e-15:
            break
        ytm_guess -= (price - market_price) / deriv * freq
        if abs(price - market_price) < 0.0001:
            break

    return round(ytm_guess * 100, 4)


def years_to_maturity_from_date(mat_date_str: str) -> float:
    """Parse a maturity date string and return years to maturity from today."""
    formats = ("%Y-%m-%d", "%m/%d/%Y", "%d/%m/%Y", "%Y/%m/%d", "%d-%b-%Y", "%d-%b-%y")
    for fmt in formats:
        try:
            mat_date = datetime.strptime(str(mat_date_str).strip(), fmt).date()
            return max((mat_date - date.today()).days / 365.25, 0.01)
        except (ValueError, TypeError):
            continue
    return 0.0


def infer_maturity_bucket(years: float) -> str:
    """Map years to maturity into a bucket."""
    if years <= 3:
        return "1-3y"
    if years <= 5:
        return "3-5y"
    if years <= 7:
        return "5-7y"
    if years <= 10:
        return "7-10y"
    return "10y+"


def resolve_columns(df: pd.DataFrame) -> dict:
    """Map DataFrame column names to canonical names using aliases."""
    col_map = {}
    df_cols_lower = {c.lower().strip().replace(" ", "_"): c for c in df.columns}
    for canonical, aliases in COLUMN_ALIASES.items():
        for alias in aliases:
            if alias in df_cols_lower:
                col_map[canonical] = df_cols_lower[alias]
                break
    return col_map


def safe_numeric(row, col_map, field, default=None):
    """Extract a numeric value from a row, returning default on failure."""
    if field not in col_map:
        return default
    val = pd.to_numeric(row.get(col_map[field], None), errors="coerce")
    return float(val) if pd.notna(val) else default


def estimate_oas(ytm_val: float, rating: str, years_to_mat: float) -> float:
    """Estimate OAS from YTM and rating when not provided directly."""
    rf_proxy = 4.0 if years_to_mat <= 5 else (4.2 if years_to_mat <= 10 else 4.5)
    ytm_based = (ytm_val - rf_proxy) * 100
    rating_based = RATING_SPREAD_MAP.get(rating, 80)
    return round(max(ytm_based, rating_based * 0.5, 10), 1)
