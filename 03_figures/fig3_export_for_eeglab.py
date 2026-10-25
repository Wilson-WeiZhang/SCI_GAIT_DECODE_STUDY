"""Figure 3 — Gait-phase locking of the imagined-walking representation

Export the Fig. 3 panel b and g maps for EEGLAB topoplot rendering

Renders one panel or table; it performs no
preprocessing and fits no model.
"""

from pathlib import Path
import sys
import numpy as np
import scipy.io as sio

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))
import fig3_harmonic_phase_responder_split as M

grp = M.load_groups()
sessions, channels, fracs, harms, A = M.load_dump()
resp = np.array([grp.get(s, (False, False, np.nan))[0] for s in sessions])
mask_full = np.array([s in grp for s in sessions])
mask_resp = mask_full & resp
mask_non = mask_full & ~resp

full_c = M.group_mean(A, mask_full)
flat = full_c.ravel()
ref_phase = np.angle(flat[np.nanargmax(np.abs(flat))])
full_signed = np.real(full_c * np.exp(-1j * ref_phase))
signed_sess = np.real(A * np.exp(-1j * ref_phase))
clim_abs = float(np.nanquantile(np.abs(full_signed), 0.98))
print("ref_phase=%.6f  shared_clim=+/-%.6f  cohort n=%d  responders n=%d  non n=%d"
      % (ref_phase, clim_abs, mask_full.sum(), mask_resp.sum(), mask_non.sum()))

cmap = M.eeglab_classic_redblue_cmap(256)(np.linspace(0, 1, 256))[:, :3]

out = dict(channels=np.array(channels, dtype=object), fracs=np.asarray(fracs, float),
           harms=np.asarray(harms, float), clim_abs=clim_abs, ref_phase=ref_phase, cmap=cmap,
           cohort_signed=full_signed, cohort_sig=M.sig_mask(signed_sess, mask_full).astype(np.uint8),
           n_cohort=int(mask_full.sum()), n_resp=int(mask_resp.sum()), n_non=int(mask_non.sum()))
for name, mask, key in [("responders", mask_resp, "resp"), ("non-responders", mask_non, "non")]:
    gc = M.group_mean(A, mask)
    out[key + "_signed"] = np.real(gc * np.exp(-1j * ref_phase))
    out[key + "_sig"] = M.sig_mask(signed_sess, mask).astype(np.uint8)
    out[key + "_mean_abs"] = np.array([float(np.nanmean(np.abs(gc[:, :, hi])))
                                       for hi in range(len(harms))])
out["cohort_mean_abs"] = np.array([float(np.nanmean(np.abs(full_c[:, :, hi])))
                                   for hi in range(len(harms))])

ki = int(np.where(np.isclose(np.asarray(harms, float), 1.0))[0][0])
k1_c = full_c[:, :, ki]
k1_flat = k1_c.ravel()
ref_k1 = np.angle(k1_flat[np.nanargmax(np.abs(k1_flat))])
k1_signed_all = np.real(k1_c * np.exp(-1j * ref_k1))
clim_k1 = float(np.nanquantile(np.abs(k1_signed_all), 0.98))
ci = int(np.argmin(np.abs(np.asarray(fracs, float) - 0.25)))
out["s6_signed"] = k1_signed_all[:, ci]
out["s6_ref_phase"] = ref_k1
out["s6_clim_abs"] = clim_k1
out["s6_cycle_fraction"] = float(fracs[ci])
print("k1-only ref_phase=%.6f  clim=+/-%.6f  cycle_fraction=%.4f"
      % (ref_k1, clim_k1, float(fracs[ci])))

dst = ROOT / "04_source_data" / "fig3" / "fig3ch_eeglab_maps.mat"
sio.savemat(dst, out)
print("wrote", dst)

from safe_write import safe_write_csv
_fr = np.asarray(fracs, float)
_g_pts = [int(np.argmin(np.abs(_fr - f))) for f in [0.0, 0.25, 0.50, 0.75, 1.0]]
_rows = []
for _map, _key, _pts in [("cohort", "cohort", range(len(_fr))),
                         ("responder", "resp", _g_pts), ("non_responder", "non", _g_pts)]:
    _v, _s = out[_key + "_signed"], out[_key + "_sig"]
    for _hi, _h in enumerate(harms):
        for _pi in _pts:
            for _ci, _ch in enumerate(channels):
                _rows.append((_map, int(_h), float(_fr[_pi]), _ch, float(_v[_ci, _pi, _hi]),
                              int(_s[_ci, _pi, _hi])))
_sd = dst.with_name("fig3ch_eeglab_maps_source_data.csv")
safe_write_csv(_sd, ["map", "harmonic_k", "cycle_fraction", "channel", "value", "significant"], _rows)
safe_write_csv(dst.with_name("fig3ch_eeglab_maps_metadata.csv"), ["key", "value"], [
    ("panel", "Fig. 3 panels b and g - gait-phase locking topographies (b: cohort, k1-k3 x 9 gait-cycle "
              "points; g: responder and non-responder sessions, k1-k3 x 5 gait-cycle points)"),
    ("producer", "03_figures/fig3_export_for_eeglab.py"),
    ("input_maps", M.DUMP.relative_to(ROOT).as_posix()),
    ("input_split", M.DEF.relative_to(ROOT).as_posix()),
    ("n_cohort", int(mask_full.sum())),
    ("n_responder", int(mask_resp.sum())),
    ("n_non_responder", int(mask_non.sum())),
    ("n_channels", len(channels)),
    ("col_map", "cohort (all sessions, Fig. 3b), responder or non_responder (Fig. 3g)"),
    ("col_harmonic_k", "gait harmonic k (1 = stride frequency, 2 = step frequency, 3 = third harmonic)"),
    ("col_cycle_fraction", "gait-cycle phase point (fraction of the cycle)"),
    ("col_channel", "EEG channel label"),
    ("col_value", "signed phase modulation: real part of the group-mean complex harmonic phase map "
                  "projected on the common cohort reference phase (one shared colour scale)"),
    ("col_significant", "1 = filled dot: one-sample t test across the group's sessions that the signed "
                        "value differs from 0, BH-FDR alpha 0.05 across channels and the nine phase points "
                        "within each harmonic"),
])
print("wrote", _sd, "(%d rows)" % len(_rows))
print("  cohort_signed", full_signed.shape, "(channels x cycle points x harmonics)")
print("  significant cells: cohort %d, responders %d, non %d"
      % (out["cohort_sig"].sum(), out["resp_sig"].sum(), out["non_sig"].sum()))
