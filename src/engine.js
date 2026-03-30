/**
 * Browser-side Black-Litterman engine.
 * Used as a fallback when the Python backend is unavailable.
 */
import * as math from "mathjs";
import { SECTORS, RATING_ORDER } from "./constants";

// ── Covariance Matrix ──

export function generateCovarianceMatrix(bonds) {
  const n = bonds.length;
  const sectorMap = {};
  SECTORS.forEach((s, i) => sectorMap[s] = i);
  const cov = math.zeros(n, n)._data;

  for (let i = 0; i < n; i++) {
    for (let j = i; j < n; j++) {
      const bi = bonds[i], bj = bonds[j];
      const volI = (bi.oas / 10000) * bi.spreadDur * 0.3;
      const volJ = (bj.oas / 10000) * bj.spreadDur * 0.3;
      let corr = 0.15;
      if (bi.sector === bj.sector) corr = 0.55;
      const ratingDist = Math.abs(RATING_ORDER.indexOf(bi.rating) - RATING_ORDER.indexOf(bj.rating));
      corr += Math.max(0, 0.15 - ratingDist * 0.03);
      if (i === j) corr = 1.0;
      corr = Math.min(corr, 0.95);
      cov[i][j] = volI * volJ * corr;
      cov[j][i] = cov[i][j];
    }
  }
  return cov;
}

function matInverse(m) {
  try { return math.inv(m); }
  catch {
    const reg = m.map((row, i) => row.map((v, j) => v + (i === j ? 1e-6 : 0)));
    return math.inv(reg);
  }
}

// ── Equilibrium Returns ──

export function computeEquilibrium(bonds, covMatrix, delta) {
  const totalMV = bonds.reduce((s, b) => s + b.mktValue, 0);
  const wMkt = bonds.map(b => b.mktValue / totalMV);
  const pi = math.multiply(math.multiply(delta, covMatrix), wMkt);
  return { wMkt, pi };
}

// ── View Matrices ──

export function buildViewMatrices(views, bonds, covMatrix, tau) {
  if (!views.length) return null;
  const n = bonds.length;
  const k = views.length;
  const P = math.zeros(k, n)._data;
  const Q = [];

  views.forEach((v, vi) => {
    Q.push(v.magnitude / 10000);
    if (v.type === "ABSOLUTE") {
      const idx = bonds.findIndex(b => b.id === v.longAssets[0]);
      if (idx >= 0) P[vi][idx] = 1;
    } else {
      const longs = v.longAssets.filter(a => bonds.some(b => b.id === a));
      const shorts = v.shortAssets.filter(a => bonds.some(b => b.id === a));
      longs.forEach(a => { const idx = bonds.findIndex(b => b.id === a); if (idx >= 0) P[vi][idx] = 1 / longs.length; });
      shorts.forEach(a => { const idx = bonds.findIndex(b => b.id === a); if (idx >= 0) P[vi][idx] = -1 / shorts.length; });
    }
  });

  const omega = math.zeros(k, k)._data;
  for (let i = 0; i < k; i++) {
    const pRow = P[i];
    const pSigPt = math.multiply(math.multiply(pRow, covMatrix), pRow);
    const conf = views[i].confidence || 0.5;
    omega[i][i] = tau * pSigPt * (1 / conf - 1);
    if (omega[i][i] < 1e-12) omega[i][i] = 1e-12;
  }
  return { P, Q, omega };
}

// ── Posterior Computation ──

export function computePosterior(pi, covMatrix, tau, viewData) {
  const tauSigma = math.multiply(tau, covMatrix);
  const tauSigmaInv = matInverse(tauSigma);

  if (!viewData) {
    return { muBL: Array.from(pi), sigmaBL: math.add(covMatrix, tauSigma) };
  }

  const { P, Q, omega } = viewData;
  const omegaInv = matInverse(omega);
  const Pt = math.transpose(P);
  const PtOmegaInvP = math.multiply(math.multiply(Pt, omegaInv), P);
  const posteriorPrecision = math.add(tauSigmaInv, PtOmegaInvP);
  const posteriorCov = matInverse(posteriorPrecision);
  const term1 = math.multiply(tauSigmaInv, pi);
  const term2 = math.multiply(math.multiply(Pt, omegaInv), Q);
  const muBL = math.multiply(posteriorCov, math.add(term1, term2));
  const sigmaBL = math.add(covMatrix, posteriorCov);

  return { muBL: Array.isArray(muBL) ? muBL : muBL._data || [muBL], sigmaBL };
}

// ── Portfolio Optimisation (gradient projection) ──

export function optimisePortfolio(muBL, sigmaBL, delta, bonds, constraints) {
  const n = bonds.length;
  let w = muBL.map(() => 1 / n);
  const lr = 0.0005;

  for (let iter = 0; iter < 5000; iter++) {
    const sigW = math.multiply(sigmaBL, w);
    const grad = math.subtract(muBL, math.multiply(delta, sigW));
    w = w.map((wi, i) => wi + lr * grad[i]);

    const maxW = (constraints.maxIssuerWeight || 5) / 100;
    const minW = (constraints.minPositionSize || 0) / 100;
    w = w.map(wi => Math.max(minW, Math.min(maxW, wi)));
    const sum = w.reduce((a, b) => a + b, 0);
    w = w.map(wi => wi / sum);
  }
  return w;
}

// ── Risk Analytics ──

export function computeRiskMetrics(w, bonds, covMatrix, muBL) {
  const portReturn = math.dot(w, muBL) * 10000;
  const portVar = math.multiply(math.multiply(w, covMatrix), w);
  const portVol = Math.sqrt(portVar) * 10000;
  const sharpe = portVol > 0 ? portReturn / portVol : 0;

  const totalMV = bonds.reduce((s, b) => s + b.mktValue, 0);
  const wBmk = bonds.map(b => b.mktValue / totalMV);
  const activeW = w.map((wi, i) => wi - wBmk[i]);
  const te = Math.sqrt(math.multiply(math.multiply(activeW, covMatrix), activeW)) * 10000;
  const ir = te > 0 ? (portReturn - math.dot(wBmk, muBL) * 10000) / te : 0;

  const el = math.dot(w, bonds.map(b => b.pd * b.lgd)) * 10000;
  const dts = math.dot(w, bonds.map(b => (b.oas / 10000) * b.spreadDur)) * 10000;
  const portSpreadDur = math.dot(w, bonds.map(b => b.spreadDur));

  const sectorWeights = {};
  const sectorBmkWeights = {};
  bonds.forEach((b, i) => {
    sectorWeights[b.sector] = (sectorWeights[b.sector] || 0) + w[i];
    sectorBmkWeights[b.sector] = (sectorBmkWeights[b.sector] || 0) + wBmk[i];
  });

  const ratingWeights = {};
  const ratingBmkWeights = {};
  bonds.forEach((b, i) => {
    const bucket = b.rating.startsWith("AAA") ? "AAA" : b.rating.startsWith("AA") ? "AA" : b.rating.startsWith("A") ? "A" : "BBB";
    ratingWeights[bucket] = (ratingWeights[bucket] || 0) + w[i];
    ratingBmkWeights[bucket] = (ratingBmkWeights[bucket] || 0) + wBmk[i];
  });

  const z99 = 2.326;
  const creditVaR = portReturn - z99 * portVol;

  return { portReturn, portVol, sharpe, te, ir, el, dts, portSpreadDur, sectorWeights, sectorBmkWeights, ratingWeights, ratingBmkWeights, creditVaR, activeW, wBmk };
}
