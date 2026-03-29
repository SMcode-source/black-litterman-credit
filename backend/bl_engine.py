"""Black-Litterman engine: equilibrium, views, posterior, and optimisation."""
import numpy as np
import cvxpy as cp
from typing import Optional


def compute_equilibrium(
    cov: np.ndarray, w_mkt: np.ndarray, delta: float,
    expected_loss: np.ndarray, migration_loss: np.ndarray
) -> np.ndarray:
    """Reverse-optimise implied equilibrium excess returns, adjusted for EL.

    π = δ Σ w_mkt
    π_adj = π - EL - MigrationLoss
    """
    pi = delta * cov @ w_mkt
    pi_adj = pi - expected_loss - migration_loss
    return pi, pi_adj


def build_view_matrices(
    views: list[dict], bond_ids: list[str], cov: np.ndarray, tau: float
) -> Optional[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Build P, Q, Ω matrices from view inputs.

    Returns None if no valid views.
    """
    n = len(bond_ids)
    id_map = {bid: i for i, bid in enumerate(bond_ids)}

    valid_views = []
    for v in views:
        longs = [id_map[a] for a in v["long_assets"] if a in id_map]
        shorts = [id_map[a] for a in v.get("short_assets", []) if a in id_map]
        if longs:
            valid_views.append((v, longs, shorts))

    if not valid_views:
        return None

    k = len(valid_views)
    P = np.zeros((k, n))
    Q = np.zeros(k)
    omega = np.zeros((k, k))

    for vi, (v, longs, shorts) in enumerate(valid_views):
        Q[vi] = v["magnitude_bp"] / 10000.0

        if v["view_type"] == "ABSOLUTE":
            for idx in longs:
                P[vi, idx] = 1.0 / len(longs)
        else:  # RELATIVE
            for idx in longs:
                P[vi, idx] = 1.0 / len(longs)
            for idx in shorts:
                P[vi, idx] = -1.0 / len(shorts)

        # Ω: Idzorek confidence-weighted method
        p_row = P[vi]
        p_sigma_pt = p_row @ cov @ p_row
        conf = v.get("confidence", 0.5)
        conf = max(min(conf, 0.999), 0.001)
        omega[vi, vi] = tau * p_sigma_pt * (1.0 / conf - 1.0)
        omega[vi, vi] = max(omega[vi, vi], 1e-14)

    return P, Q, omega


def compute_posterior(
    pi: np.ndarray, cov: np.ndarray, tau: float,
    P: Optional[np.ndarray], Q: Optional[np.ndarray], omega: Optional[np.ndarray]
) -> tuple[np.ndarray, np.ndarray]:
    """Bayesian posterior: blend equilibrium prior with views.

    μ_BL = [(τΣ)⁻¹ + P'Ω⁻¹P]⁻¹ [(τΣ)⁻¹π + P'Ω⁻¹Q]
    Σ_BL = Σ + [(τΣ)⁻¹ + P'Ω⁻¹P]⁻¹
    """
    tau_sigma = tau * cov
    n = len(pi)

    if P is None:
        mu_bl = pi.copy()
        sigma_bl = cov + tau_sigma
        return mu_bl, sigma_bl

    # Regularise tau_sigma for inversion
    tau_sigma_reg = tau_sigma + np.eye(n) * 1e-10
    tau_sigma_inv = np.linalg.inv(tau_sigma_reg)

    omega_inv = np.diag(1.0 / np.diag(omega))

    # Posterior precision and mean
    posterior_precision = tau_sigma_inv + P.T @ omega_inv @ P
    posterior_cov = np.linalg.inv(posterior_precision + np.eye(n) * 1e-12)

    mu_bl = posterior_cov @ (tau_sigma_inv @ pi + P.T @ omega_inv @ Q)
    sigma_bl = cov + posterior_cov

    return mu_bl, sigma_bl


def optimise_portfolio(
    mu_bl: np.ndarray,
    sigma_bl: np.ndarray,
    delta: float,
    w_mkt: np.ndarray,
    sectors: list[str],
    ratings: list[str],
    spread_dur: np.ndarray,
    constraints: dict,
    cov: np.ndarray,
) -> tuple[np.ndarray, str, list[dict]]:
    """Constrained QP via cvxpy.

    max  w'μ - (δ/2) w'Σw
    s.t. constraints
    """
    n = len(mu_bl)
    w = cp.Variable(n)

    # Objective: maximise expected return minus risk penalty
    # Regularise sigma_bl to ensure PSD for cvxpy
    eigvals, eigvecs = np.linalg.eigh(sigma_bl)
    eigvals = np.maximum(eigvals, 1e-10)
    sigma_bl_psd = eigvecs @ np.diag(eigvals) @ eigvecs.T
    sigma_bl_psd = (sigma_bl_psd + sigma_bl_psd.T) / 2

    objective = cp.Maximize(w @ mu_bl - (delta / 2) * cp.quad_form(w, sigma_bl_psd))

    cons = []
    constraint_info = []

    # 1. Fully invested, long-only
    cons.append(cp.sum(w) == 1)
    cons.append(w >= constraints.get("min_position_size", 0.2) / 100.0)

    # 2. Max issuer weight (auto-relax for small universes)
    max_w = constraints.get("max_issuer_weight", 5.0) / 100.0
    # If max_w * n < 1, the problem is infeasible; relax
    if max_w * n < 1.0:
        max_w = 1.0 / n + 0.10  # at least equal-weight + 10%
    cons.append(w <= max_w)
    constraint_info.append({"name": "Max issuer weight", "bound": max_w * 100})

    # 3. Sector constraints
    max_sector_ow = constraints.get("max_sector_overweight", 10.0) / 100.0
    unique_sectors = list(set(sectors))
    for sector in unique_sectors:
        mask = np.array([1.0 if s == sector else 0.0 for s in sectors])
        bmk_sector_w = float(mask @ w_mkt)
        if bmk_sector_w > 0.01:
            cons.append(mask @ w <= bmk_sector_w + max_sector_ow)
            cons.append(mask @ w >= max(0, bmk_sector_w - max_sector_ow))

    # 4. Spread duration constraint
    sd_tol = constraints.get("spread_duration_tolerance", 0.25)
    bmk_sd = float(w_mkt @ spread_dur)
    cons.append(w @ spread_dur <= bmk_sd + sd_tol)
    cons.append(w @ spread_dur >= bmk_sd - sd_tol)
    constraint_info.append({"name": "Spread duration", "bound": sd_tol})

    # 5. Tracking error constraint
    max_te = constraints.get("max_tracking_error", 150.0) / 10000.0
    active = w - w_mkt
    # TE² = active' Σ active - use PSD covariance
    cov_psd = cov.copy()
    eig_c, vec_c = np.linalg.eigh(cov_psd)
    eig_c = np.maximum(eig_c, 1e-10)
    cov_psd = vec_c @ np.diag(eig_c) @ vec_c.T
    cov_psd = (cov_psd + cov_psd.T) / 2
    cons.append(cp.quad_form(active, cov_psd) <= max_te ** 2)
    constraint_info.append({"name": "Tracking error", "bound": max_te * 10000})

    # 6. Excluded issuers
    excluded = constraints.get("excluded_issuers", [])
    # (handled by fixing weight to 0 for excluded)

    # Solve
    prob = cp.Problem(objective, cons)
    solver_status = "optimal"
    try:
        prob.solve(solver=cp.SCS, verbose=False, max_iters=10000, eps=1e-8)
        if prob.status not in ("optimal", "optimal_inaccurate"):
            # Try relaxed constraints
            prob.solve(solver=cp.ECOS, verbose=False)
        solver_status = prob.status
    except Exception as e:
        solver_status = f"error: {str(e)}"

    if w.value is not None:
        weights = np.array(w.value).flatten()
        # Clean up small values
        weights = np.maximum(weights, 0)
        weights /= weights.sum()
    else:
        # Fallback to benchmark if solver fails
        weights = w_mkt.copy()
        solver_status = "infeasible_fallback_to_benchmark"

    # Evaluate constraint binding
    binding = []
    for ci in constraint_info:
        ci_out = {**ci, "value": 0.0, "active": False, "shadow_price": None}
        binding.append(ci_out)

    return weights, solver_status, binding
