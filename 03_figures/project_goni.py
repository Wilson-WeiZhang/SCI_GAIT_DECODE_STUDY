"""Shared figure settings

Shared goniometer conventions for figure rendering

Imported by the figure scripts; it renders no panel of its own.
"""

from __future__ import annotations

DISPLAY_POLARITY = {"RHipW X": -1.0, "LKneW X": -1.0, "LAnkW X": -1.0}

_SHORT = {"LHip": "LHipW X", "LKne": "LKneW X", "LAnk": "LAnkW X",
          "RHip": "RHipW X", "RKne": "RKneW X", "RAnk": "RAnkW X"}


def polarity_vector(labels):
    """+1/-1 per label, accepting either 'LHipW X' or the short 'LHip' form."""
    out = []
    for l in labels:
        key = _SHORT.get(str(l), str(l))
        out.append(DISPLAY_POLARITY.get(key, 1.0))
    return out
