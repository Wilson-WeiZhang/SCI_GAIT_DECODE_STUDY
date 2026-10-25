"""Figure 4 — Shared locomotor representation across imagined and real walking

Healthy IW vs RW representational similarity per harmonic k1/k2/k3 (electrode RDMs of Fig. 4g)

Reads:
  decoder output in data/decoder_output/ (not distributed)

Renders one panel or table from the inputs above; it performs no
preprocessing and does not fit the decoder. It fits a least-squares regression
for the statistics reported with that panel or table.
"""

from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import linear_sum_assignment
import mne

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "data/decoder_output"
IWF = R / "diag_harmonic_phase_responder_split/persession_harmonic_phase_cycle9_complex.csv"
RWF = R / "diag_real_walking_persession/persession_harmonic_phase_cycle9_complex.csv"
REF = R / "diag_rest_yoked/persession_harmonic_phase_cycle9_complex.csv"
N_SPIN = 1000
RNG = np.random.default_rng(0)
COMMON = sorted(set(pd.read_csv(IWF).channel) & set(pd.read_csv(RWF).channel) & set(pd.read_csv(REF).channel))


def load(path, k):
    d = pd.read_csv(path); d = d[d.harmonic_k == k]
    sess = sorted(d.session.unique()); cyc = sorted(d.cycle_fraction.unique())
    re = np.full((len(sess), len(cyc), len(COMMON)), np.nan); im = np.full_like(re, np.nan)
    piv = d.set_index(["session", "cycle_fraction", "channel"])
    for si, s in enumerate(sess):
        for ci, c in enumerate(cyc):
            sub = piv.loc[(s, c)]
            re[si, ci] = sub["real"].reindex(COMMON).to_numpy(); im[si, ci] = sub["imag"].reindex(COMMON).to_numpy()
    return re, im, cyc


def crossnobis(P):
    S, C, F = P.shape
    sd = np.nanstd((P - np.nanmean(P, 0)[None]).reshape(-1, F), 0) + 1e-9
    Pw = P / sd; rdm = np.zeros((C, C))
    for i in range(C):
        for j in range(i + 1, C):
            d = Pw[:, i, :] - Pw[:, j, :]
            rdm[i, j] = rdm[j, i] = float(np.mean([np.dot(d[s], np.delete(d, s, 0).mean(0)) / F for s in range(S)]))
    return rdm


def elec_rdm(re, im):
    return crossnobis(np.concatenate([re.transpose(0, 2, 1), im.transpose(0, 2, 1)], 2))


def phase_rdm(re, im):
    return crossnobis(np.concatenate([re, im], 2))


def partial_spearman(x, y, z):
    rx, ry, rz = stats.rankdata(x), stats.rankdata(y), stats.rankdata(z)
    Z = np.c_[np.ones_like(rz), rz]; res = lambda a: a - Z @ np.linalg.lstsq(Z, a, rcond=None)[0]
    return stats.spearmanr(res(rx), res(ry))[0]


def sp(a, b):
    return stats.spearmanr(a, b)[0]


def rand_rot(rng):
    Q, Rm = np.linalg.qr(rng.normal(size=(3, 3))); Q = Q @ np.diag(np.sign(np.diag(Rm)))
    if np.linalg.det(Q) < 0: Q[:, 0] = -Q[:, 0]
    return Q


montage = mne.channels.make_standard_montage("standard_1005")
pos = montage.get_positions()["ch_pos"]; xyz = np.array([pos[c] for c in COMMON]); xyz0 = xyz - xyz.mean(0)
E, C = len(COMMON), 9
iuE = np.triu_indices(E, 1); iuP = np.triu_indices(C, 1)

rows = []
SPIN = {}; RDM_ROWS = []
print(f"common channels = {E} | healthy IW={pd.read_csv(IWF).session.nunique()} RW={pd.read_csv(RWF).session.nunique()} REST={pd.read_csv(REF).session.nunique()}")
print(f"{'k':>2} | {'spatial IW-RW':>22} {'IW-REST sp':>10} | {'temporal IW-RW raw/band':>24} {'IW-REST band':>12}")
for K in (1, 2, 3):
    D = {"IW": load(IWF, K), "RW": load(RWF, K), "REST": load(REF, K)}
    cyc = D["IW"][2]
    elec = {c: elec_rdm(D[c][0], D[c][1]) for c in D}
    phase = {c: phase_rdm(D[c][0], D[c][1]) for c in D}
    phi = np.array(cyc, float); skel = (np.sin(2 * np.pi * (phi[:, None] - phi[None, :])) ** 2)[iuP]
    skel = np.round(skel, 12)

    sp_iwrw = sp(elec["IW"][iuE], elec["RW"][iuE])
    sp_iwre = sp(elec["IW"][iuE], elec["REST"][iuE])
    tp_raw = sp(phase["IW"][iuP], phase["RW"][iuP])
    tp_par = partial_spearman(phase["IW"][iuP], phase["RW"][iuP], skel)
    tp_re_par = partial_spearman(phase["IW"][iuP], phase["REST"][iuP], skel)

    spn = np.empty(N_SPIN); a = elec["IW"][iuE]
    for b in range(N_SPIN):
        Dm = np.linalg.norm(xyz0[:, None] - (xyz0 @ rand_rot(RNG).T)[None], axis=2)
        _, pm = linear_sum_assignment(Dm)
        spn[b] = sp(a, elec["RW"][np.ix_(pm, pm)][iuE])
    p_spin = (np.sum(spn >= sp_iwrw) + 1) / (N_SPIN + 1)
    SPIN[K] = spn.copy()
    for cond in ("RW", "IW", "REST"):
        M = elec[cond]
        for i, ci_ in enumerate(COMMON):
            for j, cj_ in enumerate(COMMON):
                RDM_ROWS.append((K, cond, ci_, cj_, round(float(M[i, j]), 6)))

    print(f"{K:>2} | rho={sp_iwrw:+.3f} spin={p_spin:.3f}      {sp_iwre:+.2f}     | "
          f"{tp_raw:.2f} -> {tp_par:+.2f}            {tp_re_par:+.2f}")
    rows.append({"harmonic_k": K, "spatial_iwrw_rho": round(sp_iwrw, 4), "spatial_iwrw_spin_p": round(p_spin, 4),
                 "spatial_iwrest_rho": round(sp_iwre, 4), "temporal_iwrw_raw_rho": round(tp_raw, 4),
                 "temporal_iwrw_bandremoved_rho": round(tp_par, 4), "temporal_iwrest_bandremoved_rho": round(tp_re_par, 4)})

out = ROOT / "fig4b_rsa_per_harmonic_source_data.csv (not distributed)"
pd.DataFrame(rows).to_csv(out, index=False)
print("\nwrote", out.relative_to(ROOT))

sp_out = ROOT / "fig4b_rsa_per_harmonic_spin_null.csv (not distributed)"
pd.DataFrame({f"k{K}_spin_null_rho": SPIN[K] for K in (1, 2, 3)}).to_csv(sp_out, index=False)
rdm_out = ROOT / "04_source_data/fig4/fig4c_electrode_rdm_source_data.csv"
pd.DataFrame(RDM_ROWS, columns=["harmonic_k", "condition", "channel_i", "channel_j", "crossnobis_d"]).to_csv(rdm_out, index=False)
print("wrote", sp_out.relative_to(ROOT), "+", rdm_out.relative_to(ROOT))


import csv as _csv
import os as _osw
import re as _rew


def _write_meta(_path, _rows):
    _tmp = _path.parent / (_path.name + ".tmp")
    with _tmp.open("w", newline="", encoding="utf-8") as _f:
        _w = _csv.writer(_f)
        _w.writerow(["key", "value"])
        _w.writerows([(k, v) for k, v in _rows])
    _osw.replace(_tmp, _path)
    print("wrote", _path)


_dirs = [IWF.parent.name, RWF.parent.name, REF.parent.name]
_seven = {_d.endswith("") for _d in _dirs}
if len(_seven) != 1:
    raise SystemExit("INPUT CHECK: mixed-build_id inputs - "
                     + ", ".join(f"{_d}={'post' if _d.endswith('') else 'pre'}-main"
                                 for _d in _dirs)
                     + ". One cohort+version per figure; refusing to write a mixed-build_id "
                       "Source Data.")
_SEVEN = _seven.pop()
_ERA_DERIVED = (" " if all("" in _d for _d in _dirs) else "") + \
    ("" if _SEVEN else "reference")
_ERA_CANON = "main" if (_SEVEN and all("" in _d for _d in _dirs)) else _ERA_DERIVED
_n_iw = int(pd.read_csv(IWF, usecols=["session"]).session.nunique())
_n_rw = int(pd.read_csv(RWF, usecols=["session"]).session.nunique())
_n_re = int(pd.read_csv(REF, usecols=["session"]).session.nunique())
_SHARED = [
    ("producer", "03_figures/fig4b_rsa_per_harmonic_k123.py"),
    ("cohort", f"healthy; IW {_n_iw} / RW {_n_rw} / REST {_n_re} sessions"),
    ("input_iw", str(IWF.relative_to(ROOT)).replace("\\", "/")),
    ("input_rw", str(RWF.relative_to(ROOT)).replace("\\", "/")),
    ("input_rest", str(REF.relative_to(ROOT)).replace("\\", "/")),
    ("n_sessions_iw", _n_iw),
    ("n_sessions_rw", _n_rw),
    ("n_sessions_rest", _n_re),
    ("n_channels_common", E),
    ("n_cycle_points", C),
    ("n_harmonics", 3),
    ("n_spin", N_SPIN),
    ("spin_rng_seed", 0),
]
_write_meta(ROOT / "04_source_data/fig4/fig4b_rsa_per_harmonic_metadata.csv", _SHARED + [
    ("rows", "one per harmonic k"),
    ("n_rows", len(rows)),
    ("method_spatial", "electrode RDM (crossnobis), 2nd-order Spearman IW-vs-RW, spin-test "
                       "spatial null, IW-REST control"),
    ("method_temporal", "gait-phase RDM 2nd-order IW-vs-RW, raw and narrowband-removed "
                        "(partial on sin^2(2*pi*dphi)), plus REST"),
    ("companion_file", "fig4b_rsa_per_harmonic_spin_null.csv (not distributed) carries the "
                       "spin-null distributions behind the violins"),
    ("outputs", "fig4b_rsa_per_harmonic_source_data.csv (not distributed); "
                "fig4b_rsa_per_harmonic_spin_null.csv (not distributed)"),
])
_write_meta(ROOT / "04_source_data/fig4/fig4c_electrode_rdm_metadata.csv", _SHARED + [
    ("figure", "Fig. 4 panel g - electrode representational dissimilarity matrices "
               "for RW / IW / REST, per harmonic k1/k2/k3"),
    ("rows", "one per (harmonic_k, condition, channel_i, channel_j)"),
    ("n_rows", len(RDM_ROWS)),
    ("value_column", "crossnobis_d, rounded to 6 decimals"),
    ("outputs", "04_source_data/fig4/fig4c_electrode_rdm_source_data.csv"),
])
