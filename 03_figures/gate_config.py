"""Shared figure settings

The metadata rows build_id and build_id_note that the six walk-intent gate renderers write

Imported by the figure scripts; it renders no panel of its own.
"""

import os
import re

import analysis_config as ec

MAIN = ec.BUILD_ID == ec.MAIN
TAG = ec.MAIN
SC = True

BUILD_ID_NOTE_MAIN = (
    "causal patient preprocessing (01_preprocessing/causal/patient/: minimum-phase anti-aliasing and 0.1-40 Hz "
    "band-pass; bad channels, ICA and ICLabel fitted on a calibration recording and applied forward). Gate session "
    "set = the causal patient EEG with the gate event files (02_decoding/simulation/build_gate_session_set_causal.py); "
    "scores from 02_decoding/simulation/gate_scores_causal.py (the scorer of gate_scores_within_session.py, headline "
    "configuration). Gate features are band power / covariance of the cleaned EEG, independent of the decoder. "
    "The gate is trained on every event of the session and scored only on the events whose onset is at or after "
    "the session's calibration cut (02_decoding/simulation/gate_scored_events.py -> result/main/scored/).")


def scoped(p):
    """Location of a gate input or output: the scored gate stream (gate_within_session_*) is read from
    result/main/scored/ (gate_scored_events.py); every other path is unchanged (one build)."""
    d, b = os.path.split(p)
    if b.startswith("gate_within_session_") and os.path.basename(d) == "result":
        return os.path.join(d, "main", "scored", b)
    return p


_PATHS = []


def meta(rows):
    """"""
    if not MAIN:
        return rows
    out = []
    for k, v in rows:
        if k == "build_id":
            v = TAG
        elif k == "build_id_note":
            v = BUILD_ID_NOTE_MAIN
        elif isinstance(v, str):
            for rx, rep in _PATHS:
                v = rx.sub(rep, v)
        out.append((k, v))
    return out
