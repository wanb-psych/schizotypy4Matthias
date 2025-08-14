# Create a consolidated Python script for multi-epicenter NDM with network-rewiring nulls
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Network Diffusion Model (NDM) Epicenters + Network-Rewiring Nulls
=================================================================

Given:
  - brain_map: array shape (n,). Abnormality/atrophy/measure across n regions.
  - fc: array shape (n, n). Functional (or structural) connectivity.

This script:
  1) Preprocesses the FC (symmetrize, zero diagonal, keep positive weights).
  2) Computes per-node "best diffusion match" to the brain_map across a beta grid.
  3) Selects *m* epicenters with a greedy, diversity-aware rule.
  4) Builds network-rewiring nulls (degree-preserving swaps on a binary backbone
     + weight reassignment) and recomputes the same set statistic S_m for each null.
  5) Returns epicenters, S_m, and a one-sided p-value.

"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np
from numpy.linalg import eigh
from scipy.stats import pearsonr

# ------------------------ Random Generator ------------------------
_DEFAULT_SEED = 1234
_rng = np.random.default_rng(_DEFAULT_SEED)


# ------------------------ Utilities ------------------------
def _validate_inputs(brain_map: np.ndarray, fc: np.ndarray) -> None:
    if brain_map.ndim != 1:
        raise ValueError(f"brain_map must be 1D, got shape {brain_map.shape}")
    if fc.ndim != 2 or fc.shape[0] != fc.shape[1]:
        raise ValueError(f"fc must be square 2D, got shape {fc.shape}")
    if fc.shape[0] != brain_map.shape[0]:
        raise ValueError("fc and brain_map have incompatible sizes.")


def preprocess_fc(fc: np.ndarray, keep_positive: bool = True,
                  zero_diag: bool = True, symmetrize: bool = True) -> np.ndarray:
    """Symmetrize, zero diagonal, optionally drop negative weights."""
    A = fc.astype(float).copy()
    if symmetrize:
        A = 0.5 * (A + A.T)
    if zero_diag:
        np.fill_diagonal(A, 0.0)
    if keep_positive:
        A = np.where(A > 0, A, 0.0)
    return A


def zscore(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    x = np.asarray(x, float)
    mu = x.mean()
    sd = x.std(ddof=0)
    return (x - mu) / max(sd, eps)


def graph_laplacian(A: np.ndarray, normalized: bool = True, eps: float = 1e-12) -> np.ndarray:
    """Symmetric normalized Laplacian by default: L = I - D^{-1/2} A D^{-1/2}."""
    w = A.sum(axis=1)
    if normalized:
        with np.errstate(divide='ignore'):
            d_invsqrt = 1.0 / np.sqrt(np.maximum(w, eps))
        D_inv_sqrt = np.diag(d_invsqrt)
        L = np.eye(A.shape[0]) - D_inv_sqrt @ A @ D_inv_sqrt
    else:
        L = np.diag(w) - A
    return L


def diffusion_kernel(L: np.ndarray, beta: float, eig: Optional[Tuple[np.ndarray, np.ndarray]] = None) -> np.ndarray:
    """
    Return the heat kernel exp(-beta * L).
    Reuses eigendecomposition if provided: eig = (evals, evecs).
    """
    if eig is None:
        evals, evecs = eigh(L)
    else:
        evals, evecs = eig
    K = (evecs * np.exp(-beta * evals)) @ evecs.T
    return K


# ------------------------ NDM core ------------------------
@dataclass
class NDMResult:
    per_node_r: np.ndarray            # best correlation per node across beta
    per_node_best_beta_idx: np.ndarray
    per_node_best_beta: np.ndarray
    eig_cache: Tuple[np.ndarray, np.ndarray]


def ndm_per_node_best(A: np.ndarray, y: np.ndarray, beta_grid: np.ndarray,
                      normalized_lap: bool = True,
                      eig_cache: Optional[Tuple[np.ndarray, np.ndarray]] = None) -> NDMResult:
    """Compute each node's best Pearson r(y, K(beta) @ e_i) over beta."""
    L = graph_laplacian(A, normalized=normalized_lap)
    if eig_cache is None:
        eig_cache = eigh(L)
    Ks = [diffusion_kernel(L, b, eig=eig_cache) for b in beta_grid]

    n = A.shape[0]
    best_r = np.empty(n, dtype=float)
    best_k = np.empty(n, dtype=int)

    for i in range(n):
        e_i = np.zeros(n); e_i[i] = 1.0
        r_best = -np.inf
        k_best = 0
        for k, K in enumerate(Ks):
            xhat = K @ e_i
            r, _ = pearsonr(y, xhat)
            if r > r_best:
                r_best = r
                k_best = k
        best_r[i] = r_best
        best_k[i] = k_best

    return NDMResult(
        per_node_r=best_r,
        per_node_best_beta_idx=best_k,
        per_node_best_beta=np.array(beta_grid)[best_k],
        eig_cache=eig_cache
    )


# ------------------------ Diversity kernel ------------------------
def diffusion_similarity(A: np.ndarray, beta: float = 0.5,
                         normalized_lap: bool = True,
                         eig_cache: Optional[Tuple[np.ndarray, np.ndarray]] = None) -> Tuple[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """Similarity kernel for diversity penalty; normalized to [0,1]."""
    L = graph_laplacian(A, normalized=normalized_lap)
    if eig_cache is None:
        eig_cache = eigh(L)
    Ksim = diffusion_kernel(L, beta, eig=eig_cache)
    Ksim = Ksim / (np.max(np.abs(Ksim)) + 1e-12)
    return Ksim, eig_cache


# ------------------------ Greedy multi-epicenter selection ------------------------
@dataclass
class MultiEpicenterResult:
    indices: List[int]
    selected_scores: np.ndarray
    objective_Sm: float
    per_node_r: np.ndarray
    per_node_best_beta: np.ndarray
    eig_cache: Tuple[np.ndarray, np.ndarray]


def select_epicenters_greedy(A: np.ndarray, y: np.ndarray, beta_grid: np.ndarray, m: int = 3,
                             lambda_penalty: float = 0.25, sim_beta: float = 0.5,
                             normalized_lap: bool = True,
                             eig_cache: Optional[Tuple[np.ndarray, np.ndarray]] = None) -> MultiEpicenterResult:
    """
    Greedy selection of m epicenters with diversity penalty.
    score_i = best_r[i] - lambda_penalty * max_{j in selected} Ksim[i,j]
    """
    ndm = ndm_per_node_best(A, y, beta_grid, normalized_lap=normalized_lap, eig_cache=eig_cache)
    best_r = ndm.per_node_r
    eig_cache = ndm.eig_cache

    Ksim, _ = diffusion_similarity(A, beta=sim_beta, normalized_lap=normalized_lap, eig_cache=eig_cache)

    n = len(best_r)
    selected: List[int] = []
    selected_scores: List[float] = []
    max_sim = np.zeros(n)

    for _step in range(m):
        scores = best_r - lambda_penalty * max_sim
        if selected:
            scores[selected] = -np.inf
        i_star = int(np.argmax(scores))
        s_star = float(scores[i_star])
        selected.append(i_star)
        selected_scores.append(s_star)
        max_sim = np.maximum(max_sim, Ksim[:, i_star])

    Sm = float(np.sum(selected_scores))
    return MultiEpicenterResult(
        indices=selected,
        selected_scores=np.array(selected_scores, dtype=float),
        objective_Sm=Sm,
        per_node_r=best_r,
        per_node_best_beta=ndm.per_node_best_beta,
        eig_cache=eig_cache
    )


# ------------------------ Network rewiring nulls ------------------------
def binarize_by_density(A: np.ndarray, density: float) -> np.ndarray:
    """
    Build a symmetric binary adjacency with the target undirected edge density (0<density<1).
    Keeps top-M weights by absolute value (positive weights assumed after preprocessing).
    """
    if not (0.0 < density < 1.0):
        raise ValueError("density must be in (0,1).")
    n = A.shape[0]
    iu = np.triu_indices(n, k=1)
    weights = A[iu]
    m_possible = len(weights)
    m_keep = int(round(density * m_possible))
    m_keep = int(np.clip(m_keep, 0, m_possible))
    order = np.argsort(weights)[::-1]
    keep_idx = order[:m_keep]
    B = np.zeros_like(A, dtype=int)
    edges = (iu[0][keep_idx], iu[1][keep_idx])
    B[edges] = 1
    B[(edges[1], edges[0])] = 1
    np.fill_diagonal(B, 0)
    return B

def maslov_sneppen_rewire(B, n_swaps_per_edge=5, rng=None, max_tries_factor=10):
    """
    Perform Maslov–Sneppen rewiring on an undirected, unweighted adjacency matrix.

    Parameters
    ----------
    B : ndarray
        Binary adjacency matrix (n_nodes x n_nodes).
    n_swaps_per_edge : int
        Number of attempted swaps per edge.
    rng : np.random.Generator or None
        Random number generator for reproducibility.
    max_tries_factor : int
        Maximum tries per swap before giving up.

    Returns
    -------
    Bnull : ndarray
        Rewired binary adjacency matrix.
    """
    if rng is None:
        rng = np.random.default_rng()

    # Copy and remove self-loops
    B = np.array(B, dtype=int)
    np.fill_diagonal(B, 0)

    # Build sorted edge set (undirected, no duplicates)
    E_set = set()
    for i in range(B.shape[0]):
        for j in range(i+1, B.shape[1]):
            if B[i, j] == 1:
                E_set.add((i, j))

    E_list = list(E_set)
    n_edges = len(E_list)
    n_swaps = n_edges * n_swaps_per_edge
    max_tries = n_swaps * max_tries_factor

    tries = 0
    swaps_done = 0

    while swaps_done < n_swaps and tries < max_tries:
        tries += 1

        # Randomly pick two distinct edges
        (a, b), (c, d) = rng.choice(E_list, size=2, replace=False)

        # Ensure all nodes are distinct
        if len({a, b, c, d}) < 4:
            continue

        # Possible new edges (keep sorted for undirected)
        new_edges = [tuple(sorted((a, d))), tuple(sorted((c, b)))]

        # Skip if any new edge already exists
        if any(ne in E_set for ne in new_edges):
            continue

        # Remove old edges safely
        for e in [(a, b), (c, d)]:
            e_sorted = tuple(sorted(e))
            if e_sorted in E_set:
                E_set.remove(e_sorted)
                B[e_sorted] = 0
                B[e_sorted[::-1]] = 0

        # Add new edges
        for e in new_edges:
            E_set.add(e)
            B[e] = 1
            B[e[::-1]] = 1

        # Update E_list
        E_list = list(E_set)
        swaps_done += 1

    return B


def reassign_weights_by_rank(A_ref: np.ndarray, B_bin: np.ndarray) -> np.ndarray:
    """
    Assign sorted positive weights of A_ref to the edges of B_bin by rank.
    Preserves weight distribution while using topology of B_bin.
    """
    n = A_ref.shape[0]
    iu = np.triu_indices(n, k=1)
    ref_w = A_ref[iu]
    pos_mask = ref_w > 0
    sorted_w = np.sort(ref_w[pos_mask])  # ascending

    bin_mask = B_bin[iu] == 1
    m = bin_mask.sum()
    if m == 0:
        return np.zeros_like(A_ref, dtype=float)

    if m != len(sorted_w):
        # Match counts by taking top-m weights
        sorted_w = np.sort(ref_w[ref_w > 0])[-m:]

    idx_bin = np.where(bin_mask)[0]
    assign_order = idx_bin  # deterministic order
    w_assign = sorted_w[::-1]  # descending

    A_new = np.zeros_like(A_ref, dtype=float)
    A_new[iu[0][assign_order], iu[1][assign_order]] = w_assign
    A_new += A_new.T
    np.fill_diagonal(A_new, 0.0)
    return A_new


# ------------------------ Public API ------------------------

def run_multi_epicenters_with_network_nulls(brain_map, fc, m=3, beta_grid=None,
                                            n_nulls=100, swaps_per_edge=5,
                                            rng=None):
    """
    Multi-epicenter NDM with network nulls and per-node statistics.

    Parameters
    ----------
    brain_map : ndarray
        Observed brain pattern (n_nodes,)
    fc : ndarray
        Functional/structural connectivity (n_nodes, n_nodes)
    m : int
        Number of epicenters to select
    beta_grid : list or ndarray
        Diffusion rate candidates
    n_nulls : int
        Number of null networks
    swaps_per_edge : int
        Rewiring parameter
    rng : np.random.Generator or None
        Random number generator

    Returns
    -------
    dict with keys:
        epicenters, S_m, p_value, per_node_r, per_node_p
    """
    if rng is None:
        rng = np.random.default_rng()
    n_nodes = brain_map.shape[0]
    
    if beta_grid is None:
        beta_grid = [0.01, 0.05, 0.1, 0.2, 0.3]

    # Placeholder functions; replace with your NDM implementation
    def run_ndm_single_seed(seed_idx, A, beta):
        """Return predicted pattern from single-node seed."""
        y = np.zeros(n_nodes)
        y[seed_idx] = 1
        # simple diffusion: (I - beta*A)^(-1) * seed
        I = np.eye(n_nodes)
        return np.linalg.solve(I - beta*A, y)

    # --- Per-node statistics ---
    per_node_r = np.zeros(n_nodes)
    per_node_p = np.zeros(n_nodes)
    
    for i in range(n_nodes):
        # correlation across beta_grid (pick best beta)
        r_best = -np.inf
        for beta in beta_grid:
            y_pred = run_ndm_single_seed(i, fc, beta)
            r = np.corrcoef(y_pred, brain_map)[0,1]
            if r > r_best:
                r_best = r
        per_node_r[i] = r_best

        # Nulls
        null_r = np.zeros(n_nulls)
        for k in range(n_nulls):
            Bnull = maslov_sneppen_rewire(fc, n_swaps_per_edge=swaps_per_edge, rng=rng)
            r_max = -np.inf
            for beta in beta_grid:
                y_null = run_ndm_single_seed(i, Bnull, beta)
                r = np.corrcoef(y_null, brain_map)[0,1]
                if r > r_max:
                    r_max = r
            null_r[k] = r_max
        per_node_p[i] = np.mean(null_r >= per_node_r[i])

    # --- Multi-epicenter selection (greedy) ---
    # select m nodes with highest per_node_r
    epicenters = list(np.argsort(per_node_r)[-m:][::-1])
    # predicted pattern from multiple epicenters
    S_m = -np.inf
    for beta in beta_grid:
        y_pred = np.zeros(n_nodes)
        for idx in epicenters:
            y_pred += run_ndm_single_seed(idx, fc, beta)
        r = np.corrcoef(y_pred, brain_map)[0,1]
        if r > S_m:
            S_m = r

    # Network-null for multi-epicenter set
    null_Sm = np.zeros(n_nulls)
    for k in range(n_nulls):
        Bnull = maslov_sneppen_rewire(fc, n_swaps_per_edge=swaps_per_edge, rng=rng)
        r_max = -np.inf
        for beta in beta_grid:
            y_null = np.zeros(n_nodes)
            for idx in epicenters:
                y_null += run_ndm_single_seed(idx, Bnull, beta)
            r = np.corrcoef(y_null, brain_map)[0,1]
            if r > r_max:
                r_max = r
        null_Sm[k] = r_max

    p_value = np.mean(null_Sm >= S_m)

    return {
        "epicenters": epicenters,
        "S_m": S_m,
        "p_value": p_value,
        "per_node_r": per_node_r,
        "per_node_p": per_node_p
    }


def run_single_epicenter_with_network_nulls(
    brain_map: np.ndarray,
    fc: np.ndarray,
    *,
    beta_grid: np.ndarray = np.linspace(0.1, 10.0, 40),
    density: float = 0.15,
    n_nulls: int = 1000,
    swaps_per_edge: int = 5,
    normalized_lap: bool = True,
    keep_positive: bool = True,
    rng: np.random.Generator = _rng
) -> Dict[str, object]:
    brain_map = np.asarray(brain_map).ravel()
    fc = np.asarray(fc)
    _validate_inputs(brain_map, fc)

    y = zscore(brain_map)
    A = preprocess_fc(fc, keep_positive=keep_positive)

    # Empirical per-node r
    ndm = ndm_per_node_best(A, y, beta_grid, normalized_lap=normalized_lap)
    per_node_r = ndm.per_node_r
    empirical_max = float(per_node_r[np.argmax(per_node_r)])

    B = binarize_by_density(A, density=density)

    # Null distributions
    null_max = np.empty(n_nulls, dtype=float)
    null_r_all = np.empty((n_nulls, len(per_node_r)), dtype=float)

    for k in range(n_nulls):
        Bnull = maslov_sneppen_rewire(B, n_swaps_per_edge=swaps_per_edge, rng=rng)
        Anull = reassign_weights_by_rank(A, Bnull)
        ndm_null = ndm_per_node_best(Anull, y, beta_grid, normalized_lap=normalized_lap)

        null_r_all[k, :] = ndm_null.per_node_r
        null_max[k] = float(np.max(ndm_null.per_node_r))

    # p-value for global max
    pval_global = (np.sum(null_max >= empirical_max) + 1) / (n_nulls + 1)

    # per-node p-values
    per_node_p = np.array([
        (np.sum(null_r_all[:, i] >= per_node_r[i]) + 1) / (n_nulls + 1)
        for i in range(len(per_node_r))
    ])

    return {
        "epicenter": int(np.argmax(per_node_r)),
        "per_node_r": per_node_r.tolist(),
        "per_node_p": per_node_p.tolist(),
        "per_node_best_beta": ndm.per_node_best_beta.tolist(),
        "empirical_max_r": float(empirical_max),
        "p_value": float(pval_global),
        "null_max_distribution": null_max.tolist(),
        "settings": {
            "beta_grid": [float(b) for b in beta_grid],
            "density": float(density),
            "n_nulls": int(n_nulls),
            "swaps_per_edge": int(swaps_per_edge),
            "normalized_laplacian": bool(normalized_lap),
            "keep_positive": bool(keep_positive),
            "seed": _DEFAULT_SEED
        }
    }


# ------------------------ CLI ------------------------
def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="NDM epicenters with network-rewiring nulls")
    p.add_argument("--map", type=str, required=True, help="Path to brain_map.npy (shape (n,))")
    p.add_argument("--fc", type=str, required=True, help="Path to fc.npy (shape (n,n))")
    p.add_argument("--m", type=int, default=3, help="Number of epicenters for multi-epicenter run")
    p.add_argument("--beta-min", type=float, default=0.1)
    p.add_argument("--beta-max", type=float, default=10.0)
    p.add_argument("--beta-steps", type=int, default=40)
    p.add_argument("--sim-beta", type=float, default=0.5, help="Diversity kernel beta")
    p.add_argument("--lambda-penalty", type=float, default=0.25, help="Diversity penalty")
    p.add_argument("--density", type=float, default=0.15, help="Backbone density for rewiring")
    p.add_argument("--n-nulls", type=int, default=1000, help="Number of network nulls")
    p.add_argument("--swaps-per-edge", type=int, default=5, help="Maslov–Sneppen swaps per edge")
    p.add_argument("--unnormalized", action="store_true", help="Use unnormalized Laplacian")
    p.add_argument("--keep-negative", action="store_true", help="Keep negative FC weights")
    p.add_argument("--single", action="store_true", help="Run single-epicenter test instead of multi")
    p.add_argument("--save-json", type=str, default=None, help="Path to save JSON results")
    p.add_argument("--seed", type=int, default=_DEFAULT_SEED, help="Random seed")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    rng = np.random.default_rng(args.seed)

    brain_map = np.load(args.map).astype(float).ravel()
    fc = np.load(args.fc).astype(float)

    beta_grid = np.linspace(args.beta_min, args.beta_max, args.beta_steps)
    normalized_lap = not args.unnormalized
    keep_positive = not args.keep_negative

    if args.single:
        res = run_single_epicenter_with_network_nulls(
            brain_map, fc,
            beta_grid=beta_grid,
            density=args.density,
            n_nulls=args.n_nulls,
            swaps_per_edge=args.swaps_per_edge,
            normalized_lap=normalized_lap,
            keep_positive=keep_positive,
            rng=rng
        )
    else:
        res = run_multi_epicenters_with_network_nulls(
            brain_map, fc,
            m=args.m,
            beta_grid=beta_grid,
            sim_beta=args.sim_beta,
            lambda_penalty=args.lambda_penalty,
            density=args.density,
            n_nulls=args.n_nulls,
            swaps_per_edge=args.swaps_per_edge,
            normalized_lap=normalized_lap,
            keep_positive=keep_positive,
            rng=rng
        )

    print("=== NDM epicenters (network nulls) ===")
    if args.single:
        print(f"Epicenter (single): {res['epicenter']}")
        print(f"Empirical max r:   {res['empirical_max_r']:.4f}")
    else:
        print(f"Epicenters (m={len(res['epicenters'])}): {res['epicenters']}")
        print(f"S_m (set stat): {res['S_m']:.4f}")
    print(f"p-value: {res['p_value']:.6f}")
    print(f"Settings: {json.dumps(res['settings'], indent=2)}")

    if args.save_json:
        with open(args.save_json, "w", encoding="utf-8") as f:
            json.dump(res, f, indent=2)
        print(f"\nSaved results to {args.save_json}")


if __name__ == "__main__":
    main()

