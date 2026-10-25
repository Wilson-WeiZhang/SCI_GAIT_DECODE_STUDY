"""Decoder — run drivers (02_decoding/run)

Check of one basis build on the causal triples (build_basis.sh runs it last): against the basis of the same kind
built on the zero-phase preprocessing, every file must keep its metadata (condition, feature mode, band table,
harmonics, aggregation, stride, held-out session, sessions used) and its sample count, and its basis must differ
(proof that the causal triples reached the generator); eigenvalues finite and non-negative. On a pass it writes
<result>/bases/BASES_READY_<kind>, which export_features.sh requires.
Usage (from 02_decoding/): python run/check_basis.py <iw|exec|hglobal>

Part of the decoder's run chain (README.md, Decoding, Steps 07-09): it builds job lists,
inputs or merged outputs; it fits nothing.
"""

import hashlib
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
import run_paths as P
from pyport import matio

F = "cdm_o1_c0p18_causal"
ZERO_PHASE_PREP = {"iw": "result/shared_delta_pca_subjout_gpk123q_stride1_corr_main_cp_" + F,
       "exec": "result/shared_delta_pca_subjout_EXEC_gpk123q_stride1_corr_main_cp_" + F + "",
       "hglobal": "result/shared_delta_pca_hglobal_gpk123q_stride1_corr_main_cp_" + F}
MAIN = {"iw": P.BASIS_IW, "exec": P.BASIS_EXEC, "hglobal": P.BASIS_HG}
COND = {"iw": "MI", "exec": "EXEC", "hglobal": "MI"}
BANDS = {"iw": P.H_IW_BANDS, "exec": P.H_EXEC_BANDS, "hglobal": P.H_IW_BANDS}


def md5(p):
    h = hashlib.md5()
    with open(p, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()


def load(p):
    d = matio.loadmat_any(p)
    m = d["metadata"]
    s = lambda k: matio.to_str(m[k])
    used = sorted(matio.to_str(x) for x in matio.cell_list(d["used_sessions"]))
    return dict(basis=np.asarray(d["basis"], float), eig=np.asarray(d["eigenvalues"], float).ravel(),
                expl=np.asarray(d["explained"], float).ravel(), cond=s("condition"), feat=s("bp_band_feature_mode"),
                band=s("bp_band_csv"), bp=matio.vec(m["bp"]), klist=matio.vec(m["bp_band_harmonic_k_list"]),
                agg=s("aggregation_mode"), stride=float(matio.scalar(m["sample_stride"])),
                excl=s("target_session_excluded"), n_used=int(matio.scalar(m["n_sessions_used"])),
                n_samp=int(matio.scalar(m["n_total_samples"])), used=used)


def main():
    kind = sys.argv[1]
    self_test = "--self-test-" in sys.argv
    os.chdir(P.DMI)
    d15, d14 = (ZERO_PHASE_PREP[kind] if self_test else MAIN[kind]), ZERO_PHASE_PREP[kind]
    if kind == "hglobal":
        files = ["shared_pca_basis.mat"]
    else:
        files = sorted(f for f in os.listdir(d14) if f.endswith("_shared_delta_pca_lso.mat"))
        assert len(files) == 80, (d14, len(files))
    fails, lines, ratios = [], [], []
    for f in files:
        p15, p14 = os.path.join(d15, f), os.path.join(d14, f)
        if not os.path.exists(p15):
            fails.append((f, "missing"))
            continue
        a, b = load(p15), load(p14)
        lab = f[:-len("_shared_delta_pca_lso.mat")] if kind != "hglobal" else ""
        chk = {
            "condition": a["cond"] == COND[kind], "feature": a["feat"] == F, "band_csv": a["band"] == BANDS[kind],
            "bp": np.array_equal(a["bp"], [0.1, 2.0]), "k_list": np.array_equal(a["klist"], [1, 2, 3]),
            "aggregation": a["agg"] == "correlation", "stride": a["stride"] == 1.0,
            "excluded": a["excl"] == lab,
            "n_used": a["n_used"] == b["n_used"] and a["n_used"] in ((80,) if kind == "hglobal" else (78, 79)),
            "used_sessions": a["used"] == b["used"], "n_total_samples": a["n_samp"] == b["n_samp"],
            "shape": a["basis"].shape == b["basis"].shape,
            "eig": bool(np.all(np.isfinite(a["eig"])) and a["eig"].min() > -1e-9),
            "explained_sorted": bool(np.all(np.diff(a["expl"]) <= 1e-12)),
        }
        dmax = float(np.max(np.abs(a["basis"] - b["basis"]))) if chk["shape"] else np.inf
        chk["differs"] = (dmax <= 1e-8) if self_test else (dmax > 1e-8)
        t15, t14 = float(a["expl"][:30].sum()), float(b["expl"][:30].sum())
        ratios.append(t15 / t14)
        bad = [k for k, ok in chk.items() if not ok]
        if bad:
            fails.append((f, bad))
        lines.append("%s %s n_total_samples %d top30 %.4f (zero-phase %.4f) max|d basis| %.3g" % (
            lab or "healthy_global", md5(p15), a["n_samp"], t15, t14, dmax))
    print("\n".join(lines[:3] + (["..."] if len(lines) > 3 else [])))
    print("gate B %s %s: files %d, failing %d, top-30 explained / median %.4f (min %.4f max %.4f)%s" % (
        kind, "PASS" if not fails else "FAIL", len(files), len(fails), np.median(ratios), min(ratios), max(ratios),
        " [self-test on the zero-phase bases]" if self_test else ""))
    if fails:
        print(fails[:10])
        sys.exit(1)
    if not self_test:
        mk = os.path.join(P.R, "bases", "BASES_READY_" + kind)
        with open(mk + ".tmp", "w") as fh:
            fh.write("\n".join(lines) + "\n")
        os.replace(mk + ".tmp", mk)
        print("wrote", mk)


if __name__ == "__main__":
    main()
