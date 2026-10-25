"""Figure 7 — physics simulation and walk-intent gate

7 planar links: HAT (head, upper limbs, trunk and pelvis, one rigid body carrying the root DOF)

Part of the simulation and gate stage: it reads decoder output and
writes the rollout or classifier tables the Figure 7 renderers read.
"""

import numpy as np
import mujoco

H, M_BODY, G = 1.75, 75.0, 9.81
L_THIGH = 0.245 * H
L_SHANK = 0.246 * H
H_ANKLE = 0.039 * H
L_HAT = H - 0.530 * H
FOOT_LEN = 0.24
FOOT_BACK = 0.06
FOOT_HW, FOOT_HH = 0.045, 0.015
HIP_W = 0.18
Z_STAND_NOMINAL = H_ANKLE + L_SHANK + L_THIGH

M_THIGH, M_SHANK, M_FOOT = 0.100 * M_BODY, 0.0465 * M_BODY, 0.0145 * M_BODY
M_HAT = M_BODY - 2 * (M_THIGH + M_SHANK + M_FOOT)
COM_THIGH, COM_SHANK = 0.433 * L_THIGH, 0.433 * L_SHANK
COM_HAT = 0.626 * L_HAT
ROG = dict(thigh=0.323, shank=0.302, foot=0.475, hat=0.496)
I_THIGH = M_THIGH * (ROG["thigh"] * L_THIGH) ** 2
I_SHANK = M_SHANK * (ROG["shank"] * L_SHANK) ** 2
I_FOOT = M_FOOT * (ROG["foot"] * FOOT_LEN) ** 2
I_HAT = M_HAT * (ROG["hat"] * L_HAT) ** 2

KP = 1200.0
FORCE_CLAMP = 200.0
if __import__("os").environ.get("GAIT_F7_SIM", "causal_from2s").strip() in ("causal", "causal_from2s"):
    FORCE_CLAMP = 280.0
MU = 0.9
DT = 0.002
FS_DATA = 250.0
SETTLE_S = 0.5
KP_PITCH, KD_PITCH = 1000.0, 800.0
BWS_FRAC = 0.30
BWS_TRAVEL = 0.10
BWS_F0 = BWS_FRAC * M_BODY * G
BWS_K = BWS_F0 / BWS_TRAVEL

JRANGE = {"hip": (-45.0, 130.0), "knee": (-5.0, 145.0), "ankle": (-55.0, 35.0)}

JOINTS = ["hip_l", "knee_l", "ankle_l", "hip_r", "knee_r", "ankle_r"]
DATA_LABELS = ["LHip", "LKne", "LAnk", "RHip", "RKne", "RAnk"]
SIGN = np.array([+1.0, -1.0, +1.0, -1.0, +1.0, -1.0])


def _leg(side, y):
    s = side
    hx = f"""
      <body name="thigh_{s}" pos="0 {y:.4f} 0">
        <joint name="hip_{s}" type="hinge" axis="0 -1 0" range="{np.radians(JRANGE['hip'][0]):.4f} {np.radians(JRANGE['hip'][1]):.4f}"/>
        <inertial pos="0 0 {-COM_THIGH:.4f}" mass="{M_THIGH:.4f}" diaginertia="{I_THIGH:.5f} {I_THIGH:.5f} {0.4 * I_THIGH:.5f}"/>
        <geom name="g_thigh_{s}" type="capsule" fromto="0 0 0 0 0 {-L_THIGH:.4f}" size="0.055"/>
        <site name="s_knee_{s}" pos="0 0 {-L_THIGH:.4f}"/>
        <body name="shank_{s}" pos="0 0 {-L_THIGH:.4f}">
          <joint name="knee_{s}" type="hinge" axis="0 1 0" range="{np.radians(JRANGE['knee'][0]):.4f} {np.radians(JRANGE['knee'][1]):.4f}"/>
          <inertial pos="0 0 {-COM_SHANK:.4f}" mass="{M_SHANK:.4f}" diaginertia="{I_SHANK:.5f} {I_SHANK:.5f} {0.4 * I_SHANK:.5f}"/>
          <geom name="g_shank_{s}" type="capsule" fromto="0 0 0 0 0 {-L_SHANK:.4f}" size="0.045"/>
          <site name="s_ankle_{s}" pos="0 0 {-L_SHANK:.4f}"/>
          <body name="foot_{s}" pos="0 0 {-L_SHANK:.4f}">
            <joint name="ankle_{s}" type="hinge" axis="0 -1 0" range="{np.radians(JRANGE['ankle'][0]):.4f} {np.radians(JRANGE['ankle'][1]):.4f}"/>
            <inertial pos="{FOOT_LEN / 2 - FOOT_BACK:.4f} 0 {-H_ANKLE + FOOT_HH:.4f}" mass="{M_FOOT:.4f}" diaginertia="{I_FOOT:.5f} {I_FOOT:.5f} {I_FOOT:.5f}"/>
            <geom name="g_foot_{s}" type="box" pos="{FOOT_LEN / 2 - FOOT_BACK:.4f} 0 {-H_ANKLE + FOOT_HH:.4f}" size="{FOOT_LEN / 2:.4f} {FOOT_HW:.4f} {FOOT_HH:.4f}"/>
            <site name="s_heel_{s}" pos="{-FOOT_BACK:.4f} 0 {-H_ANKLE:.4f}"/>
            <site name="s_toe_{s}" pos="{FOOT_LEN - FOOT_BACK:.4f} 0 {-H_ANKLE:.4f}"/>
          </body>
        </body>
      </body>"""
    return hx


def _xml(kv):
    """kv = dict joint -> damping gain of its position actuator."""
    acts = "\n".join(
        f'    <position name="a_{j}" joint="{j}" kp="{KP:.1f}" kv="{kv[j]:.4f}" '
        f'forcerange="{-FORCE_CLAMP:.1f} {FORCE_CLAMP:.1f}" ctrlrange="-3.2 3.2"/>'
        for j in JOINTS)
    return f"""<mujoco model="planar_biped">
  <compiler angle="radian" inertiafromgeom="false"/>
  <option timestep="{DT}" gravity="0 0 -{G}" integrator="implicitfast"/>
  <default>
    <geom friction="{MU} 0.005 0.0001" rgba="0.62 0.64 0.70 1"/>
  </default>
  <worldbody>
    <geom name="floor" type="plane" pos="0 0 0" size="40 3 0.1" friction="{MU} 0.005 0.0001" rgba="0.85 0.85 0.85 1"/>
    <body name="torso" pos="0 0 {Z_STAND_NOMINAL:.4f}">
      <joint name="rootx" type="slide" axis="1 0 0"/>
      <joint name="rootz" type="slide" axis="0 0 1"/>
      <joint name="rootpitch" type="hinge" axis="0 -1 0"/>
      <inertial pos="0 0 {COM_HAT:.4f}" mass="{M_HAT:.4f}" diaginertia="{I_HAT:.5f} {I_HAT:.5f} {0.35 * I_HAT:.5f}"/>
      <geom name="g_hat" type="capsule" fromto="0 0 0.03 0 0 {L_HAT:.4f}" size="0.085"/>
      <site name="s_hip" pos="0 0 0"/>
      <site name="s_head" pos="0 0 {L_HAT:.4f}"/>{_leg('l', +HIP_W / 2)}{_leg('r', -HIP_W / 2)}
    </body>
  </worldbody>
  <actuator>
{acts}
  </actuator>
</mujoco>
"""


def _critical_kv():
    """Velocity gains at the critical-damping scale, computed from the model's own joint-space inertia at one
    fixed nominal pose, so the rig is identical for every rollout."""
    m = mujoco.MjModel.from_xml_string(_xml({j: 0.0 for j in JOINTS}))
    d = mujoco.MjData(m)
    d.qpos[:] = 0.0
    d.qpos[m.joint("rootz").qposadr[0]] = 0.0
    mujoco.mj_forward(m, d)
    Mfull = np.zeros((m.nv, m.nv))
    try:
        mujoco.mj_fullM(m, d, Mfull)
    except TypeError:
        mujoco.mj_fullM(m, Mfull, d.qM)
    out = {}
    for j in JOINTS:
        dof = m.joint(j).dofadr[0]
        out[j] = float(2.0 * np.sqrt(KP * Mfull[dof, dof]))
    return out


KV = _critical_kv()
MJCF = _xml(KV)


def build():
    """Return a fresh (model, data) pair for one rollout."""
    m = mujoco.MjModel.from_xml_string(MJCF)
    return m, mujoco.MjData(m)


def rig_report():
    lines = [
        f"rig: planar 7-link biped, Winter H={H} m M={M_BODY} kg, mu={MU}, dt={DT} s, "
        f"integrator=implicitfast",
        f"  segments  thigh {L_THIGH:.4f} m  shank {L_SHANK:.4f} m  ankle-height {H_ANKLE:.4f} m  "
        f"HAT {L_HAT:.4f} m  foot plate {FOOT_LEN:.2f} m",
        f"  masses    HAT {M_HAT:.3f}  thigh {M_THIGH:.3f}  shank {M_SHANK:.4f}  foot {M_FOOT:.4f} kg",
        f"  actuators kp={KP:.0f} N*m/rad, torque clamp +-{FORCE_CLAMP:.0f} N*m, "
        f"kv (critical, model-derived) " + " ".join(f"{j}={KV[j]:.2f}" for j in JOINTS),
        f"  L1' crutch-grade trunk-pitch PD kp={KP_PITCH:.0f} kd={KD_PITCH:.0f} (RUN LEVEL);  "
        f"L2' = L1' + pelvic spring F0={BWS_F0:.1f} N ({BWS_FRAC:.0%} BW) k={BWS_K:.0f} N/m, "
        f"force >= 0;  L0 = no support (diagnostic reference only)",
        f"  joint stops (deg) hip {JRANGE['hip']}  knee {JRANGE['knee']}  ankle {JRANGE['ankle']}",
        f"  sign map (data->model) {dict(zip(DATA_LABELS, SIGN.astype(int).tolist()))} "
        f"(hip flex +, knee flex +, ankle dorsi +)",
        "  NO x-direction force or velocity at any support level.",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    print(rig_report())
