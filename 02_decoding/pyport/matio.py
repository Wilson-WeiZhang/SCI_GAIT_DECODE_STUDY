"""Decoder — Python readout (02_decoding/pyport)

Reader for MATLAB .mat files (-v7.3 through h5py, -v7 through scipy.io)

Part of the decoder: it reads the per-session feature export written by
core/run_decoder.m and fits and tests the readout on it.
"""

import numpy as np


def _attr_str(obj, name):
    v = obj.attrs.get(name)
    if v is None:
        return None
    if isinstance(v, bytes):
        return v.decode()
    if isinstance(v, np.ndarray):
        v = v.ravel()[0] if v.size else b''
        return v.decode() if isinstance(v, bytes) else str(v)
    return str(v)


def _read_h5(f, obj):
    import h5py
    if isinstance(obj, h5py.Group):
        mcls = _attr_str(obj, 'MATLAB_class')
        out = {}
        for k in obj.keys():
            if k.startswith('#'):
                continue
            out[k] = _read_h5(f, obj[k])
        if mcls not in (None, 'struct'):
            out['__class__'] = mcls
        return out
    mcls = _attr_str(obj, 'MATLAB_class')
    raw = obj[()]
    if obj.attrs.get('MATLAB_empty', 0):
        if mcls == 'cell':
            return np.empty((0, 0), dtype=object)
        if mcls == 'char':
            return ''
        return np.zeros((0, 0))
    if mcls == 'char':
        a = np.asarray(raw)
        if a.ndim == 2 and a.shape[1] > 1 and a.shape[0] > 1:
            a = a.T
            return [''.join(chr(c) for c in row) for row in a]
        return ''.join(chr(c) for c in a.ravel())
    if mcls == 'cell' or (raw.dtype == object and mcls is None):
        a = np.asarray(raw)
        out = np.empty(a.shape, dtype=object)
        for idx in np.ndindex(a.shape):
            out[idx] = _read_h5(f, f[a[idx]])
        return out.T if out.ndim >= 2 else out
    a = np.asarray(raw)
    if a.dtype.names is not None and 'real' in a.dtype.names:
        a = a['real'] + 1j * a['imag']
    if mcls == 'logical':
        a = a.astype(bool)
    if a.ndim >= 2:
        a = a.T
    return a


def loadmat_any(path):
    """Read a whole .mat file into a dict of python objects (MATLAB orientation)."""
    import h5py
    try:
        is_h5 = h5py.is_hdf5(path)
    except Exception:
        is_h5 = False
    if is_h5:
        with h5py.File(path, 'r') as f:
            return {k: _read_h5(f, f[k]) for k in f.keys() if not k.startswith('#')}
    import scipy.io
    d = scipy.io.loadmat(path, squeeze_me=False, struct_as_record=False)
    return {k: v for k, v in d.items() if not k.startswith('__')}


def load_vars(path, names):
    """Read only the named top-level variables of a -v7.3 file (falls back to a full read)."""
    import h5py
    if h5py.is_hdf5(path):
        with h5py.File(path, 'r') as f:
            return {k: _read_h5(f, f[k]) for k in names if k in f}
    d = loadmat_any(path)
    return {k: d[k] for k in names if k in d}


def h5_keys(path):
    import h5py
    with h5py.File(path, 'r') as f:
        return [k for k in f.keys() if not k.startswith('#')]


def scalar(v):
    """MATLAB scalar (any wrapping) -> python float."""
    a = np.asarray(v, dtype=float).ravel()
    return float(a[0])


def vec(v):
    return np.asarray(v, dtype=float).ravel()


def cell_list(v):
    """MATLAB cell (object array, any orientation) -> flat python list in MATLAB linear order (column-major)."""
    if isinstance(v, list):
        return v
    a = np.asarray(v, dtype=object)
    return list(a.ravel(order='F'))


def to_str(v):
    if isinstance(v, str):
        return v
    a = np.asarray(v)
    if a.dtype.kind in 'US':
        return str(a.ravel()[0]) if a.size else ''
    if a.dtype == object:
        return to_str(a.ravel()[0]) if a.size else ''
    return ''.join(chr(int(c)) for c in a.ravel())


def get_field(d, name):
    """Field access for both h5 dicts and scipy mat_struct objects; tolerant of 1x1 wrapping."""
    if isinstance(d, np.ndarray) and d.dtype == object and d.size == 1:
        d = d.ravel()[0]
    if isinstance(d, dict):
        return d[name]
    return getattr(d, name)


def has_field(d, name):
    if isinstance(d, np.ndarray) and d.dtype == object and d.size == 1:
        d = d.ravel()[0]
    if isinstance(d, dict):
        return name in d
    return hasattr(d, name)
