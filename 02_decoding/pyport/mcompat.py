"""Decoder — Python readout (02_decoding/pyport)

MATLAB semantics the readout reproduces exactly: rounding, random permutations, colon ranges, fold assignment, std, corr

Part of the decoder: it reads the per-session feature export written by
core/run_decoder.m and fits and tests the readout on it.
"""

import numpy as np

EPS = float(np.finfo(float).eps)


def mround(x):
    """MATLAB round: half away from zero (np.round is banker's rounding -> wrong lag set)."""
    x = np.asarray(x, dtype=float)
    return np.sign(x) * np.floor(np.abs(x) + 0.5)


def colon(a, d, b):
    """MATLAB a:d:b for (integer-valued) doubles."""
    a, d, b = float(a), float(d), float(b)
    if d == 0 or (b - a) / d < 0:
        return np.zeros(0)
    n = int(np.floor((b - a) / d + 1e-10)) + 1
    return a + d * np.arange(n)


def parse_grid(spec):
    """'-2000:50:0' | '[1e-3 1e-2 0.1]' | list/tuple of numbers -> 1-D float array (MATLAB order kept)."""
    if isinstance(spec, (list, tuple, np.ndarray)):
        return np.asarray(spec, dtype=float).ravel()
    s = str(spec).strip()
    if s.startswith('[') and s.endswith(']'):
        s = s[1:-1]
    parts = s.replace(',', ' ').split()
    out = []
    for p in parts:
        if ':' in p:
            q = [float(v) for v in p.split(':')]
            out.extend(colon(q[0], 1.0, q[1]) if len(q) == 2 else colon(q[0], q[1], q[2]))
        else:
            out.append(float(p))
    return np.asarray(out, dtype=float)


def rand_twister(seed, n):
    """MATLAB rng(seed) (twister); rand(1, n)."""
    return np.random.RandomState(int(seed)).random_sample(int(n))


def randperm(seed, n):
    """MATLAB rng(seed); randperm(n)  -> 1-based permutation (verified against MATLAB)."""
    return np.argsort(rand_twister(seed, n), kind='stable') + 1


def fold_ids(seed, n, K):
    """fold(perm(k:K:end)) = k for k = 1..K, with perm = randperm(n) after rng(seed). Returns 1-based ids (n,)."""
    perm = randperm(seed, n)
    fold = np.zeros(int(n), dtype=np.int64)
    for k in range(1, int(K) + 1):
        fold[perm[k - 1::int(K)] - 1] = k
    return fold


def mstd(x, axis=None):
    """MATLAB std(x, 0, ...) on finite data (N-1; N == 1 -> 0)."""
    x = np.asarray(x, dtype=float)
    n = x.shape[axis] if axis is not None else x.size
    if n == 1:
        return np.zeros_like(np.mean(x, axis=axis))
    return np.std(x, axis=axis, ddof=1)


def nanmean_cols(Y):
    """MATLAB mean(Y, 1, 'omitnan'); all-NaN column -> NaN (no warning)."""
    Y = np.asarray(Y, dtype=float)
    ok = ~np.isnan(Y)
    cnt = ok.sum(axis=0)
    s = np.where(ok, Y, 0.0).sum(axis=0)
    with np.errstate(invalid='ignore', divide='ignore'):
        return np.where(cnt > 0, s / np.maximum(cnt, 1), np.nan)


def nanstd_cols(Y):
    """MATLAB std(Y, 0, 1, 'omitnan') column by column (two-pass, N-1; N == 1 -> 0; N == 0 -> NaN)."""
    Y = np.asarray(Y, dtype=float)
    out = np.full(Y.shape[1], np.nan)
    for j in range(Y.shape[1]):
        v = Y[:, j]
        v = v[~np.isnan(v)]
        if v.size == 1:
            out[j] = 0.0
        elif v.size > 1:
            out[j] = np.std(v, ddof=1)
    return out


def nanmean_vec(v):
    """MATLAB mean(v, 'omitnan') on a vector: all-NaN -> NaN."""
    v = np.asarray(v, dtype=float).ravel()
    v = v[~np.isnan(v)]
    return float(np.mean(v)) if v.size else float('nan')


def mcorr(x, y):
    """MATLAB corr(x, y) Pearson for two column vectors (centre, dot, divide by norms, clip)."""
    n = x.size
    x0 = x - np.sum(x) / n
    y0 = y - np.sum(y) / n
    c = float(x0 @ y0) / np.linalg.norm(x0) / np.linalg.norm(y0)
    if abs(c) > 1:
        c = c / abs(c)
    return c


def safe_corr(y, yp):
    """As in core/run_decoder.m, safe_corr_local."""
    valid = np.isfinite(y) & np.isfinite(yp)
    if valid.sum() < 10:
        return float('nan')
    a, b = y[valid], yp[valid]
    if mstd(a) == 0 or mstd(b) == 0:
        return float('nan')
    return mcorr(a, b)


def first_nanargmin(v):
    """MATLAB [~, i] = min(v): first minimum, NaN ignored, all-NaN -> 0 (0-based)."""
    v = np.asarray(v, dtype=float)
    if np.all(np.isnan(v)):
        return 0
    return int(np.nanargmin(v))


def first_nanargmax(v):
    v = np.asarray(v, dtype=float)
    if np.all(np.isnan(v)):
        return 0
    return int(np.nanargmax(v))
