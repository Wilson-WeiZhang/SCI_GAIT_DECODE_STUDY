"""Shared figure settings

Patient session display labels — ONE source for every figure that names a patient session

Imported by the figure scripts; it renders no panel of its own.
"""


SESSION_LABEL = {
    "pat01_sess01": ("P1", "pre 1"),
    "pat01_sess02": ("P1", "pre 2"),
    "pat01_sess04": ("P1", "post 1"),
    "pat01_sess05": ("P1", "post 2"),
    "pat01_sess06": ("P1", "post 3"),
    "pat01_sess07": ("P1", "post 4"),
    "pat01_sess08": ("P1", "post 5"),
    "pat01_sess09": ("P1", "post 6"),
    "pat02_sess01": ("P2", "pre 1"),
    "pat02_sess02": ("P2", "pre 2"),
    "pat02_sess03": ("P2", "post 1"),
    "pat02_sess05": ("P2", "post 2"),
    "pat02_sess06": ("P2", "post 3"),
    "pat02_sess08": ("P2", "post 4"),
    "pat02_sess09": ("P2", "post 5"),
}


def label(session, with_patient=False):
    """'post 3' — or 'P1 post 3' with the patient prefix."""
    pat, lab = SESSION_LABEL[session]
    return f"{pat} {lab}" if with_patient else lab
