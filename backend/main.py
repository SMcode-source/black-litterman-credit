"""FastAPI application: Black-Litterman Credit Portfolio Optimiser."""
import io
import os

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
from utils import (
    compute_ytm, years_to_maturity_from_date, infer_maturity_bucket,
    resolve_columns, safe_numeric, estimate_oas,
    RATING_PD_MAP, COLUMN_ALIASES,
)

ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "https://black-litterman-credit.onrender.com").split(",")

app = FastAPI(
    title="Black-Litterman Credit Optimiser API",
    version="1.0.0",
    description="Portfolio optimisation engine for IG credit using the Black-Litterman framework.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
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
        try:
            raw_id = str(row.get(col_map.get("id", ""), "")).strip()
            raw_issuer = str(row.get(col_map.get("issuer", ""), "")).strip()
            if not raw_id and not raw_issuer:
                continue
            bond_id = raw_id if raw_id and raw_id != "nan" else raw_issuer[:8].upper().replace(" ", "")
            issuer = raw_issuer if raw_issuer and raw_issuer != "nan" else raw_id

            sector = str(row.get(col_map.get("sector", ""), "Other")).strip()
            if sector in ("nan", ""):
                sector = "Other"

            rating = str(row.get(col_map.get("rating", ""), "")).strip().upper()
            if rating in ("NAN", ""):
                rating = "BBB"
                warnings.append(f"Row {row_num}: No rating, defaulting to BBB.")

            oas_val = safe_numeric(row, col_map, "oas", 0.0)
            if oas_val is not None and oas_val <= 0:
                oas_val = 0.0
            sd_val = safe_numeric(row, col_map, "spread_duration", 0.0)
            if sd_val is not None and sd_val <= 0:
                sd_val = 0.0
            coupon = safe_numeric(row, col_map, "coupon_rate")
            face_val = safe_numeric(row, col_map, "face_value") or 100.0
            mkt_price = safe_numeric(row, col_map, "market_price")

            # Maturity date → years to maturity
            years_to_mat = 0.0
            mat_date_str = None
            if "maturity_date" in col_map:
                mat_date_str = str(row.get(col_map["maturity_date"], "")).strip()
                if mat_date_str and mat_date_str != "nan":
                    years_to_mat = years_to_maturity_from_date(mat_date_str)
                else:
                    mat_date_str = None

            # YTM: use provided or calculate from coupon + price + maturity
            ytm_val = safe_numeric(row, col_map, "ytm")
            if ytm_val is None and coupon is not None and mkt_price is not None and years_to_mat > 0:
                ytm_val = compute_ytm(face_val, mkt_price, coupon, years_to_mat)
                if ytm_val > 0:
                    warnings.append(f"Row {row_num} ({bond_id}): YTM calculated as {ytm_val:.2f}%.")

            # Estimate spread duration from maturity if missing
            if sd_val <= 0 and years_to_mat > 0:
                sd_val = round(years_to_mat * 0.85, 2)
                warnings.append(f"Row {row_num} ({bond_id}): Spread duration estimated as {sd_val:.1f}y.")

            # Estimate OAS from YTM/rating if missing
            if oas_val <= 0 and ytm_val is not None:
                oas_val = estimate_oas(ytm_val, rating, years_to_mat)
                warnings.append(f"Row {row_num} ({bond_id}): OAS estimated as {oas_val:.0f} bp.")

            # Maturity bucket
            mat_bucket = None
            if "maturity_bucket" in col_map:
                mb = str(row.get(col_map["maturity_bucket"], "")).strip()
                mat_bucket = mb if mb and mb != "nan" else None
            if not mat_bucket:
                mat_bucket = infer_maturity_bucket(years_to_mat) if years_to_mat > 0 else "3-5y"

            # Apply safe defaults for missing fields
            if oas_val <= 0 and sd_val <= 0:
                errors.append(f"Row {row_num} ({bond_id}): Cannot determine OAS or spread duration. Skipping.")
                continue
            if oas_val <= 0:
                oas_val = 50
                warnings.append(f"Row {row_num} ({bond_id}): OAS defaulted to 50 bp.")
            if sd_val <= 0:
                sd_val = 4.0
                warnings.append(f"Row {row_num} ({bond_id}): Spread duration defaulted to 4.0y.")

            pd_val = safe_numeric(row, col_map, "pd_annual")
            if pd_val is None or pd_val <= 0:
                pd_val = RATING_PD_MAP.get(rating, 0.003)
            lgd_val = safe_numeric(row, col_map, "lgd")
            if lgd_val is None or not (0 <= lgd_val <= 1):
                lgd_val = 0.4
            mv_val = safe_numeric(row, col_map, "mkt_value")
            if mv_val is None or mv_val <= 0:
                mv_val = 1000.0

            parsed_bonds.append(ParsedBond(
                id=bond_id, issuer=issuer, sector=sector, rating=rating,
                oas=oas_val, spread_duration=sd_val, pd_annual=pd_val,
                lgd=lgd_val, mkt_value=mv_val, maturity_bucket=mat_bucket,
                coupon_rate=coupon, maturity_date=mat_date_str,
                face_value=face_val, market_price=mkt_price, ytm=ytm_val,
            ))
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
