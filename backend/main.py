"""FastAPI application: Black-Litterman Credit Portfolio Optimiser."""
import io
import math
from datetime import datetime, date

import numpy as np
import pandas as pd
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from models import (
    OptimisationRequest, OptimisationResponse, IssuerResult,
    RiskMetrics, SectorAllocation, RatingAllocation, ConstraintStatus,
    ParsedBond, UploadResponse,
)
from risk_engine import (
    build_covariance_matrix, compute_expected_losses,
    compute_migration_loss, monte_carlo_credit_var,
)
from bl_engine import (
    compute_equilibrium, build_view_matrices,
    compute_posterior, optimise_portfolio,
)

app = FastAPI(
    title="Black-Litterman Credit Optimiser API",
    version="1.0.0",
    description="Portfolio optimisation engine for IG credit using the Black-Litterman framework.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    return {"status": "ok", "engine": "bl-credit-optimiser", "version": "1.0.0"}


@app.post("/api/v1/optimise", response_model=OptimisationResponse)
def run_optimisation(req: OptimisationRequest):
    warnings = []
    n = len(req.bonds)

    # ── Extract arrays ──
    bond_ids = [b.id for b in req.bonds]
    issuers = [b.issuer for b in req.bonds]
    sectors = [b.sector for b in req.bonds]
    ratings = [b.rating for b in req.bonds]
    mat_buckets = [b.maturity_bucket for b in req.bonds]
    oas = np.array([b.oas for b in req.bonds])
    spread_dur = np.array([b.spread_duration for b in req.bonds])
    pd_arr = np.array([b.pd_annual for b in req.bonds])
    lgd_arr = np.array([b.lgd for b in req.bonds])
    mkt_val = np.array([b.mkt_value for b in req.bonds])

    # Validate
    if np.any(oas <= 0):
        bad = [bond_ids[i] for i in range(n) if oas[i] <= 0]
        warnings.append(f"Non-positive OAS for: {bad}. Excluded from optimisation.")

    # ── Benchmark weights ──
    w_mkt = mkt_val / mkt_val.sum()

    # ── Covariance matrix ──
    cov = build_covariance_matrix(oas, spread_dur, sectors, ratings, mat_buckets, req.covariance_method)

    # ── Expected losses & migration ──
    el = compute_expected_losses(pd_arr, lgd_arr)
    mig_loss = compute_migration_loss(ratings, spread_dur)

    # ── Step 1: Equilibrium returns ──
    pi_raw, pi_adj = compute_equilibrium(cov, w_mkt, req.delta, el, mig_loss)

    # ── Step 2-3: Views and posterior ──
    views_dicts = [
        {
            "view_type": v.view_type,
            "long_assets": v.long_assets,
            "short_assets": v.short_assets,
            "magnitude_bp": v.magnitude_bp,
            "confidence": v.confidence,
        }
        for v in req.views
    ]

    view_result = build_view_matrices(views_dicts, bond_ids, cov, req.tau)
    if view_result is not None:
        P, Q, omega = view_result
        mu_bl, sigma_bl = compute_posterior(pi_adj, cov, req.tau, P, Q, omega)
    else:
        mu_bl, sigma_bl = compute_posterior(pi_adj, cov, req.tau, None, None, None)
        warnings.append("No valid views provided. Using equilibrium returns only.")

    # ── Step 4: Optimisation ──
    constraints_dict = {
        "max_issuer_weight": req.constraints.max_issuer_weight,
        "min_position_size": req.constraints.min_position_size,
        "max_sector_overweight": req.constraints.max_sector_overweight,
        "max_tracking_error": req.constraints.max_tracking_error,
        "spread_duration_tolerance": req.constraints.spread_duration_tolerance,
        "excluded_issuers": req.constraints.excluded_issuers,
    }

    w_opt, solver_status, constraint_binding = optimise_portfolio(
        mu_bl, sigma_bl, req.delta, w_mkt, sectors, ratings,
        spread_dur, constraints_dict, cov
    )

    # ── Risk metrics ──
    port_ret = float(w_opt @ mu_bl) * 10000
    port_var = float(w_opt @ cov @ w_opt)
    port_vol = np.sqrt(port_var) * 10000
    sharpe = port_ret / port_vol if port_vol > 0 else 0

    active_w = w_opt - w_mkt
    te = np.sqrt(float(active_w @ cov @ active_w)) * 10000
    bmk_ret = float(w_mkt @ mu_bl) * 10000
    ir = (port_ret - bmk_ret) / te if te > 0 else 0

    port_el = float(w_opt @ el) * 10000
    port_sd = float(w_opt @ spread_dur)
    port_dts = float(w_opt @ (oas / 10000.0 * spread_dur)) * 10000

    # ── Monte Carlo Credit VaR ──
    mc_result = monte_carlo_credit_var(
        w_opt, pd_arr, lgd_arr, ratings, spread_dur, sectors,
        n_sim=req.n_simulations
    )

    risk_metrics = RiskMetrics(
        portfolio_return_bp=round(port_ret, 2),
        portfolio_vol_bp=round(port_vol, 2),
        sharpe_ratio=round(sharpe, 4),
        tracking_error_bp=round(te, 2),
        information_ratio=round(ir, 4),
        expected_loss_bp=round(port_el, 2),
        spread_duration=round(port_sd, 3),
        dts=round(port_dts, 2),
        credit_var_99_bp=round(mc_result["var_99"] * 10000, 2),
        credit_cvar_99_bp=round(mc_result["cvar_99"] * 10000, 2),
    )

    # ── Issuer results ──
    issuer_results = []
    for i in range(n):
        issuer_results.append(IssuerResult(
            id=bond_ids[i],
            issuer=issuers[i],
            sector=sectors[i],
            rating=ratings[i],
            benchmark_weight=round(float(w_mkt[i]) * 100, 4),
            optimal_weight=round(float(w_opt[i]) * 100, 4),
            active_weight=round(float(active_w[i]) * 100, 4),
            equilibrium_return_bp=round(float(pi_adj[i]) * 10000, 2),
            posterior_return_bp=round(float(mu_bl[i]) * 10000, 2),
            expected_loss_bp=round(float(el[i]) * 10000, 2),
            dts=round(float(oas[i] / 10000 * spread_dur[i]) * 10000, 2),
        ))

    # ── Sector allocation ──
    sector_alloc = []
    unique_sectors = sorted(set(sectors))
    for s in unique_sectors:
        mask = np.array([1.0 if sec == s else 0.0 for sec in sectors])
        bmk_w = float(mask @ w_mkt) * 100
        opt_w = float(mask @ w_opt) * 100
        sector_alloc.append(SectorAllocation(
            sector=s, benchmark=round(bmk_w, 2),
            optimal=round(opt_w, 2), active=round(opt_w - bmk_w, 2)
        ))

    # ── Rating allocation ──
    rating_alloc = []
    for bucket_name in ["AAA", "AA", "A", "BBB"]:
        bucket_ratings = {
            "AAA": ["AAA"],
            "AA": ["AA+", "AA", "AA-"],
            "A": ["A+", "A", "A-"],
            "BBB": ["BBB+", "BBB", "BBB-"],
        }[bucket_name]
        mask = np.array([1.0 if r in bucket_ratings else 0.0 for r in ratings])
        bmk_w = float(mask @ w_mkt) * 100
        opt_w = float(mask @ w_opt) * 100
        rating_alloc.append(RatingAllocation(
            rating_bucket=bucket_name, benchmark=round(bmk_w, 2),
            optimal=round(opt_w, 2), active=round(opt_w - bmk_w, 2)
        ))

    # ── Constraint status ──
    con_status = [
        ConstraintStatus(name=c["name"], bound=c["bound"], value=0, active=False)
        for c in constraint_binding
    ]

    return OptimisationResponse(
        status="success",
        solver_status=solver_status,
        issuer_results=issuer_results,
        risk_metrics=risk_metrics,
        sector_allocation=sector_alloc,
        rating_allocation=rating_alloc,
        constraint_status=con_status,
        var_distribution=mc_result["percentiles"],
        warnings=warnings,
    )


@app.post("/api/v1/equilibrium")
def get_equilibrium(req: OptimisationRequest):
    """Return equilibrium returns without views or optimisation."""
    oas = np.array([b.oas for b in req.bonds])
    spread_dur = np.array([b.spread_duration for b in req.bonds])
    pd_arr = np.array([b.pd_annual for b in req.bonds])
    lgd_arr = np.array([b.lgd for b in req.bonds])
    mkt_val = np.array([b.mkt_value for b in req.bonds])
    sectors = [b.sector for b in req.bonds]
    ratings = [b.rating for b in req.bonds]
    mat_buckets = [b.maturity_bucket for b in req.bonds]

    w_mkt = mkt_val / mkt_val.sum()
    cov = build_covariance_matrix(oas, spread_dur, sectors, ratings, mat_buckets, req.covariance_method)
    el = compute_expected_losses(pd_arr, lgd_arr)
    mig_loss = compute_migration_loss(ratings, spread_dur)
    pi_raw, pi_adj = compute_equilibrium(cov, w_mkt, req.delta, el, mig_loss)

    return {
        "equilibrium_returns": {
            b.id: {"raw_bp": round(float(pi_raw[i]) * 10000, 2),
                   "adjusted_bp": round(float(pi_adj[i]) * 10000, 2)}
            for i, b in enumerate(req.bonds)
        }
    }


# ═══════════════════════════════════════════════════════════════════
# YIELD-TO-MATURITY CALCULATION
# ═══════════════════════════════════════════════════════════════════

def compute_ytm(face_value: float, market_price: float,
                coupon_rate: float, years_to_maturity: float,
                freq: int = 2) -> float:
    """Newton-Raphson YTM solver. Returns annual yield in percent."""
    if years_to_maturity <= 0 or market_price <= 0:
        return 0.0
    coupon = face_value * (coupon_rate / 100.0) / freq
    n_periods = int(round(years_to_maturity * freq))
    if n_periods < 1:
        n_periods = 1

    # Initial guess from current yield
    ytm_guess = (coupon * freq) / market_price

    for _ in range(200):
        r = ytm_guess / freq
        if abs(r) < 1e-12:
            r = 1e-12
        # Price as function of yield
        disc = (1 + r)
        pv_coupons = coupon * (1 - disc ** (-n_periods)) / r
        pv_face = face_value * disc ** (-n_periods)
        price = pv_coupons + pv_face
        # Derivative
        d_pv_c = -coupon / (r * freq) * (
            -(-n_periods) * disc ** (-n_periods - 1) / r * r
            - (1 - disc ** (-n_periods)) / (r)
        )
        # Simplified numerical derivative
        dr = 0.0001
        r2 = r + dr
        disc2 = (1 + r2)
        price2 = coupon * (1 - disc2 ** (-n_periods)) / r2 + face_value * disc2 ** (-n_periods)
        deriv = (price2 - price) / dr

        if abs(deriv) < 1e-15:
            break
        ytm_guess -= (price - market_price) / deriv * freq
        if abs(price - market_price) < 0.0001:
            break

    return round(ytm_guess * 100, 4)


def years_to_maturity_from_date(mat_date_str: str) -> float:
    """Parse a maturity date string and return years to maturity from today."""
    for fmt in ("%Y-%m-%d", "%m/%d/%Y", "%d/%m/%Y", "%Y/%m/%d", "%d-%b-%Y", "%d-%b-%y"):
        try:
            mat_date = datetime.strptime(str(mat_date_str).strip(), fmt).date()
            delta = (mat_date - date.today()).days
            return max(delta / 365.25, 0.01)
        except (ValueError, TypeError):
            continue
    return 0.0


def infer_maturity_bucket(years: float) -> str:
    """Map years to maturity into a bucket."""
    if years <= 3:
        return "1-3y"
    elif years <= 5:
        return "3-5y"
    elif years <= 7:
        return "5-7y"
    elif years <= 10:
        return "7-10y"
    else:
        return "10y+"


# Default PD by rating (annualised, approximate Moody's)
RATING_PD_MAP = {
    "AAA": 0.0001, "AA+": 0.0002, "AA": 0.0003, "AA-": 0.0004,
    "A+": 0.0006, "A": 0.0008, "A-": 0.0012,
    "BBB+": 0.0020, "BBB": 0.0035, "BBB-": 0.0055,
    "BB+": 0.0080, "BB": 0.0120, "BB-": 0.0180,
    "B+": 0.0300, "B": 0.0500, "B-": 0.0800,
}

# Flexible column name mapping
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


@app.post("/api/v1/upload-bonds", response_model=UploadResponse)
async def upload_bonds(file: UploadFile = File(...)):
    """Parse a CSV, TXT, or Excel file into bond universe data."""
    warnings = []
    errors = []

    # Read file content
    content = await file.read()
    filename = (file.filename or "").lower()

    try:
        if filename.endswith((".xlsx", ".xls")):
            df = pd.read_excel(io.BytesIO(content), engine="openpyxl")
        elif filename.endswith(".csv") or filename.endswith(".txt"):
            # Try comma first, then tab, then semicolon
            text = content.decode("utf-8", errors="replace")
            for sep in [",", "\t", ";", "|"]:
                df = pd.read_csv(io.StringIO(text), sep=sep)
                if len(df.columns) > 2:
                    break
        else:
            raise HTTPException(status_code=400, detail="Unsupported file type. Use .csv, .txt, .xlsx, or .xls")

        if df.empty:
            raise HTTPException(status_code=400, detail="Uploaded file contains no data rows.")

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to parse file: {str(e)}")

    # Resolve column names
    col_map = resolve_columns(df)

    # Check minimum required columns
    required_one_of = {
        "identifier": ["id", "issuer"],
        "spread_info": ["oas"],
    }
    if "id" not in col_map and "issuer" not in col_map:
        errors.append("File must have an 'id' or 'issuer' column.")
        return UploadResponse(status="error", bonds=[], row_count=0, warnings=warnings, errors=errors)

    # Parse each row
    parsed_bonds = []
    for idx, row in df.iterrows():
        row_num = idx + 2  # 1-indexed + header
        bond = {}
        try:
            # ID / Issuer
            raw_id = str(row.get(col_map.get("id", ""), "")).strip()
            raw_issuer = str(row.get(col_map.get("issuer", ""), "")).strip()
            if not raw_id and not raw_issuer:
                continue  # skip blank rows
            bond["id"] = raw_id if raw_id and raw_id != "nan" else raw_issuer[:8].upper().replace(" ", "")
            bond["issuer"] = raw_issuer if raw_issuer and raw_issuer != "nan" else raw_id

            # Sector (default "Other")
            bond["sector"] = str(row.get(col_map.get("sector", ""), "Other")).strip()
            if bond["sector"] in ("nan", ""):
                bond["sector"] = "Other"

            # Rating
            rating_raw = str(row.get(col_map.get("rating", ""), "")).strip().upper()
            if rating_raw in ("NAN", ""):
                rating_raw = "BBB"
                warnings.append(f"Row {row_num}: No rating provided, defaulting to BBB.")
            bond["rating"] = rating_raw

            # OAS
            if "oas" in col_map:
                oas_val = pd.to_numeric(row.get(col_map["oas"], None), errors="coerce")
                bond["oas"] = float(oas_val) if pd.notna(oas_val) and oas_val > 0 else 0.0
            else:
                bond["oas"] = 0.0

            # Spread duration
            if "spread_duration" in col_map:
                sd_val = pd.to_numeric(row.get(col_map["spread_duration"], None), errors="coerce")
                bond["spread_duration"] = float(sd_val) if pd.notna(sd_val) and sd_val > 0 else 0.0
            else:
                bond["spread_duration"] = 0.0

            # Coupon, maturity date, face value, market price (for YTM calc)
            coupon = None
            if "coupon_rate" in col_map:
                c = pd.to_numeric(row.get(col_map["coupon_rate"], None), errors="coerce")
                coupon = float(c) if pd.notna(c) else None
            bond["coupon_rate"] = coupon

            mat_date_str = None
            years_to_mat = 0.0
            if "maturity_date" in col_map:
                mat_date_str = str(row.get(col_map["maturity_date"], "")).strip()
                if mat_date_str and mat_date_str != "nan":
                    years_to_mat = years_to_maturity_from_date(mat_date_str)
                    bond["maturity_date"] = mat_date_str
                else:
                    mat_date_str = None

            face_val = None
            if "face_value" in col_map:
                f = pd.to_numeric(row.get(col_map["face_value"], None), errors="coerce")
                face_val = float(f) if pd.notna(f) else None
            bond["face_value"] = face_val or 100.0

            mkt_price = None
            if "market_price" in col_map:
                p = pd.to_numeric(row.get(col_map["market_price"], None), errors="coerce")
                mkt_price = float(p) if pd.notna(p) else None
            bond["market_price"] = mkt_price

            # YTM: use provided value or calculate
            ytm_val = None
            if "ytm" in col_map:
                y = pd.to_numeric(row.get(col_map["ytm"], None), errors="coerce")
                ytm_val = float(y) if pd.notna(y) else None

            if ytm_val is None and coupon is not None and mkt_price is not None and years_to_mat > 0:
                ytm_val = compute_ytm(bond["face_value"], mkt_price, coupon, years_to_mat)
                if ytm_val > 0:
                    warnings.append(f"Row {row_num} ({bond['id']}): YTM calculated as {ytm_val:.2f}%.")
            bond["ytm"] = ytm_val

            # Spread duration: if not provided, estimate from maturity
            if bond["spread_duration"] <= 0 and years_to_mat > 0:
                # Modified duration approximation
                bond["spread_duration"] = round(years_to_mat * 0.85, 2)
                warnings.append(f"Row {row_num} ({bond['id']}): Spread duration estimated from maturity as {bond['spread_duration']:.1f}y.")

            # OAS: if not provided but YTM and benchmark info available, estimate
            if bond["oas"] <= 0 and ytm_val is not None:
                # Use rating-based spread as a proxy when YTM-based estimate is poor
                rating_spread_map = {
                    "AAA": 35, "AA+": 45, "AA": 50, "AA-": 55,
                    "A+": 65, "A": 75, "A-": 85,
                    "BBB+": 105, "BBB": 130, "BBB-": 160,
                    "BB+": 220, "BB": 300, "BB-": 400,
                }
                # Try YTM-based estimate (OAS ≈ YTM - treasury proxy)
                # Use 2y≈4.5%, 5y≈4.0%, 10y≈4.2% rough proxies
                rf_proxy = 4.0 if years_to_mat <= 5 else 4.2 if years_to_mat <= 10 else 4.5
                ytm_based_oas = (ytm_val - rf_proxy) * 100
                rating_based_oas = rating_spread_map.get(bond["rating"], 80)
                # Use the larger of YTM-implied or rating-based, floor at rating/2
                estimated_oas = max(ytm_based_oas, rating_based_oas * 0.5, 10)
                bond["oas"] = round(estimated_oas, 1)
                warnings.append(f"Row {row_num} ({bond['id']}): OAS estimated as {bond['oas']:.0f} bp.")

            # Maturity bucket
            if "maturity_bucket" in col_map:
                mb = str(row.get(col_map["maturity_bucket"], "")).strip()
                bond["maturity_bucket"] = mb if mb and mb != "nan" else infer_maturity_bucket(years_to_mat)
            elif years_to_mat > 0:
                bond["maturity_bucket"] = infer_maturity_bucket(years_to_mat)
            else:
                bond["maturity_bucket"] = "3-5y"

            # PD
            if "pd_annual" in col_map:
                pd_val = pd.to_numeric(row.get(col_map["pd_annual"], None), errors="coerce")
                bond["pd_annual"] = float(pd_val) if pd.notna(pd_val) and pd_val > 0 else RATING_PD_MAP.get(bond["rating"], 0.003)
            else:
                bond["pd_annual"] = RATING_PD_MAP.get(bond["rating"], 0.003)

            # LGD
            if "lgd" in col_map:
                lgd_val = pd.to_numeric(row.get(col_map["lgd"], None), errors="coerce")
                bond["lgd"] = float(lgd_val) if pd.notna(lgd_val) and 0 <= lgd_val <= 1 else 0.4
            else:
                bond["lgd"] = 0.4

            # Market value
            if "mkt_value" in col_map:
                mv = pd.to_numeric(row.get(col_map["mkt_value"], None), errors="coerce")
                bond["mkt_value"] = float(mv) if pd.notna(mv) and mv > 0 else 1000.0
            else:
                bond["mkt_value"] = 1000.0

            # Validate essential fields
            if bond["oas"] <= 0 and bond["spread_duration"] <= 0:
                errors.append(f"Row {row_num} ({bond['id']}): Cannot determine OAS or spread duration. Skipping.")
                continue

            if bond["oas"] <= 0:
                bond["oas"] = 50  # safe default
                warnings.append(f"Row {row_num} ({bond['id']}): OAS defaulted to 50 bp.")

            if bond["spread_duration"] <= 0:
                bond["spread_duration"] = 4.0
                warnings.append(f"Row {row_num} ({bond['id']}): Spread duration defaulted to 4.0y.")

            parsed_bonds.append(ParsedBond(**bond))

        except Exception as e:
            errors.append(f"Row {row_num}: Failed to parse - {str(e)}")

    status = "success" if parsed_bonds else "error"
    if not parsed_bonds and not errors:
        errors.append("No valid bonds could be parsed from the file.")

    return UploadResponse(
        status=status,
        bonds=parsed_bonds,
        row_count=len(parsed_bonds),
        warnings=warnings[:50],  # cap warnings
        errors=errors[:50],
    )
