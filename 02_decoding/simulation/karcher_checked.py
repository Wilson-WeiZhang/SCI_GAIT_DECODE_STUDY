"""Figure 7 — physics simulation and walk-intent gate

karcher_checked — a convergence-checked AIRM (Karcher) mean solver

Part of the simulation and gate stage: it reads decoder output and
writes the rollout or classifier tables the Figure 7 renderers read.
"""

import os
import sys
from collections import namedtuple

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import gate_fusion as PF

TOL_ABS = 1e-10
TOL_REL = 1e-8
ITMAX_CHECKED = 20000
TAU0 = 1.0
TAU_MIN = 2.0 ** -10

KarcherResult = namedtuple(
    "KarcherResult",
    "mean n_iter final_norm exit_residual rel_residual converged n_halvings tau_final")


def _tangent(R, Cs):
    """(S, Rp, dispersion) with S = mean_i log(R^-1/2 C_i R^-1/2), Rp = R^1/2."""
    Rm, Rp = PF.powm_sym(R, -0.5), PF.powm_sym(R, 0.5)
    Ls = [PF.logm_sym(Rm @ C @ Rm) for C in Cs]
    S = np.mean(Ls, axis=0)
    disp = float(np.mean([np.linalg.norm(L, "fro") for L in Ls]))
    return S, Rp, disp


def karcher_residual(R, Cs):
    """||mean_i log(R^-1/2 C_i R^-1/2)||_F -- exactly zero at the AIRM (Karcher) mean."""
    Rm = PF.powm_sym(R, -0.5)
    S = np.mean([PF.logm_sym(Rm @ C @ Rm) for C in Cs], axis=0)
    return float(np.linalg.norm(S, "fro"))


def riemann_mean_frozen(Cs, tol=1e-10, itmax=60):
    """The fixed-iteration schedule of gate_fusion.riemann_mean, instrumented."""
    R = np.mean(Cs, axis=0)
    n_iter, last = 0, float("nan")
    broke = False
    for _ in range(itmax):
        n_iter += 1
        Rm, Rp = PF.powm_sym(R, -0.5), PF.powm_sym(R, 0.5)
        S = np.mean([PF.logm_sym(Rm @ C @ Rm) for C in Cs], axis=0)
        R = Rp @ PF._expm_sym(S) @ Rp
        last = float(np.linalg.norm(S, "fro"))
        if last < tol:
            broke = True
            break
    R = (R + R.T) / 2.0
    res = karcher_residual(R, Cs)
    disp = _tangent(R, Cs)[2]
    return KarcherResult(mean=R, n_iter=n_iter, final_norm=last, exit_residual=res,
                         rel_residual=res / disp if disp > 0 else float("nan"),
                         converged=bool(broke), n_halvings=0, tau_final=1.0)


def riemann_mean_checked(Cs, tol=TOL_ABS, tol_rel=TOL_REL, itmax=ITMAX_CHECKED,
                         tau0=TAU0, tau_min=TAU_MIN):
    """The same fixed-point iteration WITH a convergence test and an adaptive step size."""
    R = np.mean(Cs, axis=0)
    R = (R + R.T) / 2.0
    tau, prev, n_halv, n_iter = float(tau0), float("inf"), 0, 0
    norm = float("nan")
    for _ in range(itmax):
        n_iter += 1
        S, Rp, disp = _tangent(R, Cs)
        norm = float(np.linalg.norm(S, "fro"))
        if norm < tol:
            res = karcher_residual(R, Cs)
            return KarcherResult(mean=R, n_iter=n_iter, final_norm=norm, exit_residual=res,
                                 rel_residual=res / disp if disp > 0 else float("nan"),
                                 converged=True, n_halvings=n_halv, tau_final=tau)
        if norm > prev:
            tau = max(tau / 2.0, tau_min)
            n_halv += 1
        prev = norm
        R = Rp @ PF._expm_sym(tau * S) @ Rp
        R = (R + R.T) / 2.0
    res = karcher_residual(R, Cs)
    disp = _tangent(R, Cs)[2]
    return KarcherResult(mean=R, n_iter=n_iter, final_norm=norm, exit_residual=res,
                         rel_residual=res / disp if disp > 0 else float("nan"),
                         converged=False, n_halvings=n_halv, tau_final=tau)


def band_karcher(calib_covs, lam, mode):
    """Per band b: the Karcher mean of [shrink(c[b], lam) for c in calib_covs]."""
    solver = riemann_mean_checked if mode == "checked" else riemann_mean_frozen
    out = []
    for b in range(len(PF.BANDS)):
        Cs = [PF.shrink(c[b], lam) for c in calib_covs]
        out.append(solver(Cs))
    return out


def make_whiteners(mode, record=None):
    """A drop-in replacement for gate_fusion.whiteners(calib_covs, lam)."""
    def _whiteners(calib_covs, lam=PF.LAM_RA):
        res = band_karcher(calib_covs, lam, mode)
        if record is not None:
            record.extend(res)
        return [PF.powm_sym(r.mean, -0.5) for r in res]
    return _whiteners
