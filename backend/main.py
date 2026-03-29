"""FastAPI application: Black-Litterman Credit Portfolio Optimiser."""
import numpy as np
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from models import (
    OptimisationRequest, OptimisationResponse, IssuerResult,
    RiskMetrics, SectorAllocation, RatingAllocation, ConstraintStatus,
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
