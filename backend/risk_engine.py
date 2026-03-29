"""Risk engine: covariance estimation, default/migration risk, Credit VaR."""
import numpy as np
from scipy import linalg
from typing import Optional

RATING_ORDER = ["AAA", "AA+", "AA", "AA-", "A+", "A", "A-", "BBB+", "BBB", "BBB-"]
RATING_BUCKETS = {"AAA": 0, "AA+": 1, "AA": 1, "AA-": 1, "A+": 2, "A": 2, "A-": 2,
                  "BBB+": 3, "BBB": 3, "BBB-": 3}

# Annual transition matrix (simplified Moody's-style for IG)
# Rows/cols: AAA, AA, A, BBB, BB (HY), Default
TRANSITION_MATRIX = np.array([
    [0.9081, 0.0833, 0.0068, 0.0006, 0.0012, 0.0000],
    [0.0070, 0.9065, 0.0779, 0.0064, 0.0006, 0.0016],
    [0.0009, 0.0227, 0.9105, 0.0552, 0.0074, 0.0033],
    [0.0002, 0.0033, 0.0595, 0.8693, 0.0530, 0.0147],
    [0.0003, 0.0014, 0.0067, 0.0773, 0.8053, 0.1090],
    [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 1.0000],
])

# Typical spread change (bp) for each migration: from rating bucket -> to bucket
MIGRATION_SPREAD_CHANGE = np.array([
    [0,    -15,   -70,  -200,  -400, -600],  # from AAA
    [15,     0,   -55,  -180,  -380, -580],  # from AA
    [70,    55,     0,  -120,  -320, -520],  # from A
    [200,  180,   120,     0,  -200, -400],  # from BBB
    [400,  380,   320,   200,     0, -200],  # from BB
    [600,  580,   520,   400,   200,    0],  # from Default
])

SECTOR_LIST = [
    "Technology", "Financials", "Healthcare", "Energy",
    "Consumer Staples", "Communications", "Utilities",
    "Industrials", "Consumer Disc.", "Real Estate", "Materials"
]


def _rating_bucket_index(rating: str) -> int:
    """Map granular rating to transition matrix row index."""
    bucket = RATING_BUCKETS.get(rating, 3)
    return min(bucket, 4)


def build_factor_model_covariance(
    oas: np.ndarray,
    spread_dur: np.ndarray,
    sectors: list[str],
    ratings: list[str],
    mat_buckets: list[str],
) -> np.ndarray:
    """Build structured factor-model covariance matrix.

    Decomposes spread return volatility into systematic factors
    (market, sector, rating, curve) and idiosyncratic residual.
    Σ = B V B' + D
    """
    n = len(oas)

    # Spread return volatility for each bond: σ_i ≈ OAS_i × SD_i × vol_scalar
    # Typical IG spread vol is ~30% of OAS level annualised
    vol_scalar = 0.30
    vols = (oas / 10000.0) * spread_dur * vol_scalar

    # Factor loadings matrix B: [market, sector_1..K, rating_1..4, curve_1..5]
    sector_map = {s: i for i, s in enumerate(SECTOR_LIST)}
    n_sectors = len(SECTOR_LIST)
    rating_bucket_map = {"AAA": 0, "AA": 1, "A": 2, "BBB": 3}
    n_rating_factors = 4
    bucket_map = {"1-3y": 0, "3-5y": 1, "5-7y": 2, "7-10y": 3, "10y+": 4}
    n_curve_factors = 5
    n_factors = 1 + n_sectors + n_rating_factors + n_curve_factors

    B = np.zeros((n, n_factors))
    for i in range(n):
        # Market factor loading (beta ≈ 1 for all)
        B[i, 0] = 1.0
        # Sector factor
        si = sector_map.get(sectors[i], 0)
        B[i, 1 + si] = 0.7
        # Rating factor
        rb = RATING_BUCKETS.get(ratings[i], 3)
        B[i, 1 + n_sectors + rb] = 0.5
        # Curve factor
        cb = bucket_map.get(mat_buckets[i], 1)
        B[i, 1 + n_sectors + n_rating_factors + cb] = 0.3

    # Factor covariance V
    # Market factor vol ~ 50bp annualised for IG
    mkt_var = (0.0050) ** 2
    V = np.eye(n_factors) * (0.0015) ** 2  # base small variance
    V[0, 0] = mkt_var
    for j in range(1, 1 + n_sectors):
        V[j, j] = (0.0025) ** 2  # sector factor vol
    for j in range(1 + n_sectors, 1 + n_sectors + n_rating_factors):
        V[j, j] = (0.0020) ** 2  # rating factor vol
    for j in range(1 + n_sectors + n_rating_factors, n_factors):
        V[j, j] = (0.0015) ** 2  # curve factor vol

    # Systematic covariance: B V B'
    systematic = B @ V @ B.T

    # Idiosyncratic diagonal: D
    idio_share = 0.35  # 35% of total variance is idiosyncratic
    D = np.diag((vols * np.sqrt(idio_share)) ** 2)

    # Total Σ = systematic + D, scaled by individual vols
    sigma = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            sigma[i, j] = systematic[i, j] * vols[i] * vols[j] / (
                np.sqrt(systematic[i, i] + D[i, i]) * np.sqrt(systematic[j, j] + D[j, j]) + 1e-12
            ) * vols[i] * vols[j]
    np.fill_diagonal(sigma, vols ** 2)

    # Ensure PSD via eigenvalue floor
    eigvals, eigvecs = np.linalg.eigh(sigma)
    eigvals = np.maximum(eigvals, 1e-10)
    sigma = eigvecs @ np.diag(eigvals) @ eigvecs.T
    sigma = (sigma + sigma.T) / 2

    return sigma


def build_ledoit_wolf_covariance(
    oas: np.ndarray, spread_dur: np.ndarray, sectors: list[str], ratings: list[str]
) -> np.ndarray:
    """Ledoit-Wolf shrinkage toward single-factor target."""
    n = len(oas)
    vols = (oas / 10000.0) * spread_dur * 0.30

    # Sample correlation with sector structure
    corr = np.full((n, n), 0.20)
    for i in range(n):
        for j in range(i, n):
            c = 0.20
            if sectors[i] == sectors[j]:
                c += 0.35
            ri = RATING_ORDER.index(ratings[i]) if ratings[i] in RATING_ORDER else 5
            rj = RATING_ORDER.index(ratings[j]) if ratings[j] in RATING_ORDER else 5
            c += max(0, 0.10 - abs(ri - rj) * 0.02)
            if i == j:
                c = 1.0
            corr[i, j] = min(c, 0.95)
            corr[j, i] = corr[i, j]

    # Target: single-factor model
    avg_corr = (corr.sum() - n) / (n * (n - 1))
    target = np.full((n, n), avg_corr)
    np.fill_diagonal(target, 1.0)

    # Shrinkage intensity (fixed reasonable value)
    shrinkage = 0.3
    shrunk_corr = (1 - shrinkage) * corr + shrinkage * target

    # Convert to covariance
    sigma = np.outer(vols, vols) * shrunk_corr
    return sigma


def build_covariance_matrix(
    oas: np.ndarray,
    spread_dur: np.ndarray,
    sectors: list[str],
    ratings: list[str],
    mat_buckets: list[str],
    method: str = "factor_model",
) -> np.ndarray:
    """Dispatch to the appropriate covariance estimator."""
    if method == "factor_model":
        return build_factor_model_covariance(oas, spread_dur, sectors, ratings, mat_buckets)
    elif method == "ledoit_wolf":
        return build_ledoit_wolf_covariance(oas, spread_dur, sectors, ratings)
    else:  # ewma or fallback
        return build_ledoit_wolf_covariance(oas, spread_dur, sectors, ratings)


def compute_expected_losses(pd: np.ndarray, lgd: np.ndarray) -> np.ndarray:
    """Expected loss = PD × LGD for each issuer."""
    return pd * lgd


def compute_migration_loss(ratings: list[str], spread_dur: np.ndarray) -> np.ndarray:
    """Expected spread return loss from rating migrations."""
    n = len(ratings)
    mig_loss = np.zeros(n)
    for i in range(n):
        ri = _rating_bucket_index(ratings[i])
        # Probability-weighted spread change
        for j in range(6):  # AAA, AA, A, BBB, BB, Default
            prob = TRANSITION_MATRIX[ri, j]
            spread_chg_bp = MIGRATION_SPREAD_CHANGE[ri, j]
            # Loss = spread widening × spread duration (negative = loss)
            mig_loss[i] += prob * (spread_chg_bp / 10000.0) * spread_dur[i]
    return mig_loss


def monte_carlo_credit_var(
    weights: np.ndarray,
    pd: np.ndarray,
    lgd: np.ndarray,
    ratings: list[str],
    spread_dur: np.ndarray,
    sectors: list[str],
    n_sim: int = 50000,
    confidence: float = 0.99,
    seed: int = 42,
) -> dict:
    """Monte Carlo Credit VaR via Gaussian copula.

    Returns VaR, CVaR, and loss distribution percentiles.
    """
    rng = np.random.default_rng(seed)
    n = len(weights)

    # Build asset correlation matrix from sector structure
    corr = np.eye(n)
    for i in range(n):
        for j in range(i + 1, n):
            c = 0.20  # base inter-sector
            if sectors[i] == sectors[j]:
                c = 0.50
            corr[i, j] = c
            corr[j, i] = c

    # Cholesky decomposition
    try:
        L = np.linalg.cholesky(corr)
    except np.linalg.LinAlgError:
        eigvals, eigvecs = np.linalg.eigh(corr)
        eigvals = np.maximum(eigvals, 1e-8)
        corr = eigvecs @ np.diag(eigvals) @ eigvecs.T
        corr = (corr + corr.T) / 2
        np.fill_diagonal(corr, 1.0)
        L = np.linalg.cholesky(corr)

    # Default thresholds from PD via normal inverse
    from scipy.stats import norm
    default_thresholds = norm.ppf(pd)

    # Migration thresholds for downgrade
    downgrade_thresholds = np.zeros(n)
    for i in range(n):
        ri = _rating_bucket_index(ratings[i])
        # Probability of downgrade by 1+ notches
        p_down = sum(TRANSITION_MATRIX[ri, ri + 1:]) if ri < 5 else 0
        downgrade_thresholds[i] = norm.ppf(max(p_down, 1e-10))

    # Simulate
    Z = rng.standard_normal((n_sim, n))
    correlated_Z = Z @ L.T

    losses = np.zeros(n_sim)
    for sim in range(n_sim):
        scenario_loss = 0.0
        for i in range(n):
            if weights[i] < 1e-8:
                continue
            z = correlated_Z[sim, i]
            if z < default_thresholds[i]:
                # Default event
                scenario_loss += weights[i] * lgd[i]
            elif z < downgrade_thresholds[i]:
                # Downgrade event: spread widening loss
                ri = _rating_bucket_index(ratings[i])
                # Assume 1-notch downgrade spread impact
                spread_impact = abs(MIGRATION_SPREAD_CHANGE[ri, min(ri + 1, 5)]) / 10000.0
                scenario_loss += weights[i] * spread_impact * spread_dur[i]
        losses[sim] = scenario_loss

    losses_sorted = np.sort(losses)
    var_idx = int(n_sim * confidence)
    var_99 = losses_sorted[min(var_idx, n_sim - 1)]
    cvar_99 = losses_sorted[var_idx:].mean() if var_idx < n_sim else var_99

    # Distribution percentiles for charting
    percentiles = [float(np.percentile(losses, p)) for p in range(0, 101, 5)]

    return {
        "var_99": float(var_99),
        "cvar_99": float(cvar_99),
        "mean_loss": float(losses.mean()),
        "percentiles": percentiles,
        "losses_histogram": [float(np.percentile(losses, p)) for p in range(1, 100)],
    }
