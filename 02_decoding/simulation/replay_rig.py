"""Figure 7 — physics simulation and walk-intent gate

replay_rig — one full-physics rollout of the biped model under a joint-angle target stream

Part of the simulation and gate stage: it reads decoder output and
writes the rollout or classifier tables the Figure 7 renderers read.
"""

import numpy as np
import mujoco

import biped_model as BM

FALL_PITCH_RAD = np.radians(60.0)
FALL_HEIGHT_FRAC = 0.5
STEP_MIN_ADVANCE = 0.01
PASS_MIN_STEPS = 6
PASS_MIN_DX = 0.3
SUBSTEPS = int(round((1.0 / BM.FS_DATA) / BM.DT))
F7_SIM = __import__("os").environ.get("GAIT_F7_SIM", "causal_from2s").strip() in ("causal", "causal_from2s")
WIN0_S = 1.0
_REZERO = F7_SIM and __import__("os").environ.get("GAIT_F7_WARM", "").strip() != "apred"
if F7_SIM and __import__("os").environ.get("GAIT_F7_INIT", "").strip():
    _REZERO, WIN0_S = True, 2.0
_INIT_REC = F7_SIM and bool(__import__("os").environ.get("GAIT_F7_INIT", "").strip())
if __import__("os").environ.get("GAIT_F7_SIM", "causal_from2s").strip() == "causal_from2s":
    _REZERO, WIN0_S = True, 2.0


def _foot_min_z(m, d, gid):
    """Lowest world z of a foot box's 8 corners."""
    c, R, s = d.geom_xpos[gid], d.geom_xmat[gid].reshape(3, 3), m.geom_size[gid]
    sgn = np.array([[a, b, cc] for a in (-1, 1) for b in (-1, 1) for cc in (-1, 1)], float)
    return float((c + (sgn * s) @ R.T)[:, 2].min())


def rollout(targets, level="L0", record_stride=25):
    """targets [T, 6] rad in actuator order.  level in {L0, L1, L2}."""
    m, d = BM.build()
    jadr = [m.joint(j).qposadr[0] for j in BM.JOINTS]
    rx, rz, rp = (m.joint(n).qposadr[0] for n in ("rootx", "rootz", "rootpitch"))
    vrx, vrz, vrp = (m.joint(n).dofadr[0] for n in ("rootx", "rootz", "rootpitch"))
    gid = {s: m.geom(f"g_foot_{s}").id for s in ("l", "r")}
    floor = m.geom("floor").id
    body_geoms = {m.geom(f"g_{n}").id for n in
                  ("hat", "thigh_l", "thigh_r", "shank_l", "shank_r")}
    sid = {n: m.site(f"s_{n}").id for n in
           ("hip", "head", "knee_l", "ankle_l", "heel_l", "toe_l",
            "knee_r", "ankle_r", "heel_r", "toe_r")}

    d.qpos[:] = 0.0
    for a, v in zip(jadr, targets[0]):
        d.qpos[a] = v
    mujoco.mj_forward(m, d)
    d.qpos[rz] -= min(_foot_min_z(m, d, gid["l"]), _foot_min_z(m, d, gid["r"]))
    d.qvel[:] = 0.0
    mujoco.mj_forward(m, d)
    z0 = float(d.site_xpos[sid["hip"], 2])
    x_ref0 = float(d.site_xpos[sid["hip"], 0])

    n_settle = int(round(BM.SETTLE_S / BM.DT))
    n_main = len(targets) * SUBSTEPS
    contact_lr = np.zeros((n_main, 2), bool)
    foot_x = np.zeros((n_main, 2))
    clamp_hits, n_force = 0, 0
    tau_pitch_int, fz_int, tau_peak = 0.0, 0.0, 0.0
    trace = []
    t_fall, fell, body_ground = np.nan, False, False
    dx, k = 0.0, 0

    def support(phase_step):
        """Apply the level's injections as generalized forces on the ROOT dofs."""
        nonlocal tau_pitch_int, fz_int, tau_peak
        d.qfrc_applied[:] = 0.0
        if level in ("L1p", "L2p"):
            tau = -BM.KP_PITCH * d.qpos[rp] - BM.KD_PITCH * d.qvel[vrp]
            d.qfrc_applied[vrp] = tau
            if phase_step:
                tau_pitch_int += abs(float(tau)) * BM.DT
                tau_peak = max(tau_peak, abs(float(tau)))
        if level == "L2p":
            fz = max(0.0, BM.BWS_F0 + BM.BWS_K * (z0 - float(d.site_xpos[sid["hip"], 2])))
            d.qfrc_applied[vrz] = fz
            if phase_step:
                fz_int += fz * BM.DT

    for _ in range(n_settle):
        d.ctrl[:] = targets[0]
        support(False)
        mujoco.mj_step(m, d)

    x0 = float(d.site_xpos[sid["hip"], 0])
    w0_steps, x_w0 = int(round(WIN0_S / BM.DT)), float("nan")
    x_w05 = float("nan")
    pitch_settle = float(np.degrees(d.qpos[rp]))
    hipz_settle = float(d.site_xpos[sid["hip"], 2])
    for f in range(len(targets)):
        d.ctrl[:] = targets[f]
        for _ in range(SUBSTEPS):
            support(True)
            mujoco.mj_step(m, d)
            af = d.actuator_force
            clamp_hits += int(np.sum(np.abs(af) >= BM.FORCE_CLAMP - 1e-6))
            n_force += 6
            for i, s in enumerate(("l", "r")):
                foot_x[k, i] = d.geom_xpos[gid[s], 0]
            for ci in range(d.ncon):
                g1, g2 = d.contact.geom1[ci], d.contact.geom2[ci]
                pair = {int(g1), int(g2)}
                if floor in pair:
                    other = (pair - {floor}).pop()
                    if other == gid["l"]:
                        contact_lr[k, 0] = True
                    elif other == gid["r"]:
                        contact_lr[k, 1] = True
                    elif other in body_geoms:
                        body_ground = True
            if _REZERO and k == w0_steps:
                x_w0 = float(d.site_xpos[sid["hip"], 0])
            if _INIT_REC and k == w0_steps // 4:
                x_w05 = float(d.site_xpos[sid["hip"], 0])
            if k % record_stride == 0:
                trace.append([(k + 1) * BM.DT] +
                             [d.site_xpos[sid[n], j] for n in sid for j in (0, 2)])
            pitch, hz = abs(float(d.qpos[rp])), float(d.site_xpos[sid["hip"], 2])
            if not fell and (pitch > FALL_PITCH_RAD or hz < FALL_HEIGHT_FRAC * z0
                             or not np.isfinite(hz)):
                fell, t_fall = True, (k + 1) * BM.DT
            k += 1
            if fell:
                break
        if fell:
            break

    n_eval = max(k, 1)
    dx = float(d.site_xpos[sid["hip"], 0] - x0)
    T_total = len(targets) / BM.FS_DATA
    survival = t_fall if fell else T_total
    streak, n_trans = _step_streak(contact_lr[:n_eval], foot_x[:n_eval])
    cl_w, fx_w, tr_w = contact_lr[:n_eval], foot_x[:n_eval], trace
    if _REZERO:
        T_total, survival = T_total - WIN0_S, max(survival - WIN0_S, 0.0)
        t_fall = t_fall - WIN0_S if fell else t_fall
        if n_eval > w0_steps:
            cl_w, fx_w = contact_lr[w0_steps:n_eval], foot_x[w0_steps:n_eval]
            dx = float(d.site_xpos[sid["hip"], 0] - x_w0)
            tr_w = [[r[0] - WIN0_S] + list(r[1:]) for r in trace if r[0] > WIN0_S]
        else:
            cl_w, fx_w, dx, tr_w = contact_lr[:0], foot_x[:0], 0.0, []
        streak, n_trans = _step_streak(cl_w, fx_w)
    passed = bool((not fell) and streak >= PASS_MIN_STEPS and dx >= PASS_MIN_DX)
    out = dict(
        level=level, T_s=T_total, n_eval_steps=n_eval,
        fell=int(fell), t_fall=t_fall, survival_s=float(survival),
        n_steps_max_streak=int(streak), n_support_transitions=int(n_trans),
        dx_m=dx, clamp_frac=float(clamp_hits / max(n_force, 1)),
        tau_int=float(tau_pitch_int),
        tau_peak=float(tau_peak),
        support_z_impulse_frac=float(fz_int / (BM.M_BODY * BM.G * max(survival, 1e-9))),
        stand_height_m=z0, body_ground=int(body_ground),
        pitch_after_settle_deg=pitch_settle, hip_z_after_settle_m=hipz_settle,
        frac_double_support=float(np.mean(cl_w.all(1))) if len(cl_w) else float("nan"),
        frac_flight=float(np.mean(~cl_w.any(1))) if len(cl_w) else float("nan"),
        final_pitch_deg=float(np.degrees(d.qpos[rp])),
        final_hip_z_m=float(d.site_xpos[sid["hip"], 2]),
        passed=int(passed), trace=np.asarray(tr_w, float),
        site_order=list(sid.keys()), x_ref0=x_ref0,
        contact_lr=cl_w, foot_x=fx_w)
    if _INIT_REC:
        out.update(warm_fell=int(fell and t_fall < 0), warm_dx_m=(x_w0 - x0) if np.isfinite(x_w0) else float("nan"),
                   warm_dx05_m=(x_w05 - x0) if np.isfinite(x_w05) else float("nan"),
                   warm_trace=np.asarray([r for r in trace if r[0] <= WIN0_S], float))
    return out


def _step_streak(contact, fx):
    """Longest run of consecutive alternating single-support transitions in which the
    incoming support foot is >= STEP_MIN_ADVANCE in front (+x) of the outgoing one."""
    single = np.where(contact.sum(1) == 1, np.argmax(contact, axis=1), -1)
    single[contact.sum(1) != 1] = -1
    epochs = []
    for i, s in enumerate(single):
        if s >= 0 and (not epochs or epochs[-1][0] != s):
            epochs.append((int(s), i))
    best = cur = 0
    n_trans = 0
    for a, b in zip(epochs, epochs[1:]):
        n_trans += 1
        adv = fx[b[1], b[0]] - fx[b[1], a[0]]
        if adv >= STEP_MIN_ADVANCE:
            cur += 1
            best = max(best, cur)
        else:
            cur = 0
    return best, n_trans


def filmstrip(res, path, title="", n_frames=8):
    """Stick-figure filmstrip from the recorded site trace (no MuJoCo renderer)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    tr = res["trace"]
    if len(tr) < 2:
        return None
    names = res["site_order"]
    idx = np.linspace(0, len(tr) - 1, min(n_frames, len(tr))).astype(int)
    P = {n: tr[:, 1 + 2 * i:3 + 2 * i] for i, n in enumerate(names)}
    fig, axes = plt.subplots(1, len(idx), figsize=(1.55 * len(idx), 3.1), sharey=True)
    axes = np.atleast_1d(axes)
    chains = [("hip", "head"), ("hip", "knee_l"), ("knee_l", "ankle_l"),
              ("ankle_l", "heel_l"), ("ankle_l", "toe_l"),
              ("hip", "knee_r"), ("knee_r", "ankle_r"),
              ("ankle_r", "heel_r"), ("ankle_r", "toe_r")]
    col = {"l": "#1f6fb4", "r": "#c8471f"}
    for ax, i in zip(axes, idx):
        xc = P["hip"][i, 0]
        ax.axhline(0, color="0.35", lw=1.2, zorder=0)
        for a, b in chains:
            c = col["l"] if a.endswith("_l") or b.endswith("_l") else (
                col["r"] if a.endswith("_r") or b.endswith("_r") else "0.15")
            ax.plot([P[a][i, 0] - xc, P[b][i, 0] - xc], [P[a][i, 1], P[b][i, 1]],
                    color=c, lw=2.6, solid_capstyle="round")
        ax.plot([0], [P["hip"][i, 1]], "o", color="0.15", ms=5)
        ax.set_xlim(-0.75, 0.75)
        ax.set_ylim(-0.08, 1.95)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.set_title(f"{tr[i, 0]:.2f} s\nx={P['hip'][i, 0]:+.2f} m", fontsize=7)
    fig.suptitle(title, fontsize=8)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(path, dpi=130, facecolor="white")
    plt.close(fig)
    return path
