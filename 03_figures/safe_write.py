#!/usr/bin/env python3
"""Shared figure settings

safe_write.py - the one way code/ scripts replace a file on disk

Imported by the figure scripts; it renders no panel of its own.
"""

from __future__ import annotations

import csv
import io
import os
import stat
import sys
import tempfile
from pathlib import Path


def safe_write(path, data, *, encoding="utf-8", newline="\n", allow_empty=False) -> Path:
    path = Path(path)
    if isinstance(data, (bytes, bytearray)):
        payload = bytes(data)
    elif isinstance(data, str):
        nl = os.linesep if newline is None else (newline or "\n")
        payload = (data.replace("\n", nl) if nl != "\n" else data).encode(encoding)
    else:
        raise TypeError("safe_write: data must be str or bytes, not %s" % type(data).__name__)

    if not payload and not allow_empty and path.exists() and path.stat().st_size > 0:
        raise ValueError("safe_write: refusing to replace non-empty %s with empty content "
                         "(pass allow_empty=True if that is intended)" % path)

    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=path.name + ".", suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as fh:
            fd = None
            fh.write(payload)
            fh.flush()
            os.fsync(fh.fileno())
        if path.exists():
            os.chmod(tmp, stat.S_IMODE(path.stat().st_mode))
        else:
            umask = os.umask(0)
            os.umask(umask)
            os.chmod(tmp, 0o666 & ~umask)
        os.replace(tmp, path)
    except BaseException:
        if fd is not None:
            os.close(fd)
        try:
            os.unlink(tmp)
        except FileNotFoundError:
            pass
        raise

    if path.read_bytes() != payload:
        raise RuntimeError("safe_write: re-read of %s does not match what was written" % path)
    return path


def safe_write_csv(path, header, rows, *, encoding="utf-8") -> Path:
    buf = io.StringIO(newline="")
    w = csv.writer(buf, lineterminator="\n")
    w.writerow(header)
    w.writerows(rows)
    return safe_write(path, buf.getvalue(), encoding=encoding, newline="\n")


def _selftest() -> int:
    results = []

    def check(name, fn):
        try:
            fn()
            results.append((name, True, ""))
        except Exception as exc:
            results.append((name, False, "%s: %s" % (type(exc).__name__, exc)))

    with tempfile.TemporaryDirectory() as d:
        d = Path(d)

        def tmp_siblings(p):
            return sorted(p.parent.glob(p.name + ".*.tmp"))

        def t_new():
            p = d / "a_new.txt"
            safe_write(p, "one\ntwo\n")
            assert p.read_bytes() == b"one\ntwo\n", p.read_bytes()

        def t_overwrite():
            p = d / "b_over.txt"
            p.write_bytes(b"old content that is longer than the new one\n")
            safe_write(p, "new\n")
            assert p.read_bytes() == b"new\n", p.read_bytes()
            assert not tmp_siblings(p), tmp_siblings(p)

        def t_empty_refused():
            p = d / "c_empty.txt"
            p.write_bytes(b"keep me\n")
            try:
                safe_write(p, "")
            except ValueError:
                pass
            else:
                raise AssertionError("empty write over a non-empty file was not refused")
            assert p.read_bytes() == b"keep me\n", p.read_bytes()
            assert not tmp_siblings(p), tmp_siblings(p)
            safe_write(p, b"", allow_empty=True)
            assert p.read_bytes() == b"", "allow_empty=True did not write the empty file"

        def t_failure_rolls_back():
            p = d / "d_fail.txt"
            p.write_bytes(b"original\n")
            real = os.replace

            def boom(*a, **k):
                raise OSError("forced failure in os.replace")

            os.replace = boom
            try:
                safe_write(p, "replacement\n")
            except OSError:
                pass
            else:
                raise AssertionError("forced failure did not propagate")
            finally:
                os.replace = real
            assert p.read_bytes() == b"original\n", p.read_bytes()
            assert not tmp_siblings(p), tmp_siblings(p)

        def t_bytes():
            p = d / "e_bytes.bin"
            data = bytes(range(256)) + b"a,b\r\nc,d\r\n"
            safe_write(p, data)
            assert p.read_bytes() == data

        def t_csv_lf():
            p = d / "f_table.csv"
            safe_write_csv(p, ["k", "v"], [["x", "1"], ["y", "a,b"], ["z", "line\nbreak"]])
            b = p.read_bytes()
            assert b"\r" not in b, b
            assert b.startswith(b"k,v\nx,1\ny,\"a,b\"\n"), b

        check("(a) new file written and re-read equal", t_new)
        check("(b) overwrite leaves exactly the new bytes, no *.tmp sibling", t_overwrite)
        check("(c) empty data on a non-empty file raises ValueError, file unchanged", t_empty_refused)
        check("(d) failure in os.replace leaves the original bytes, no temp file", t_failure_rolls_back)
        check("(e) bytes input round-trips", t_bytes)
        check("(f) safe_write_csv output is LF-only", t_csv_lf)

    for name, ok, why in results:
        print("%s  %s%s" % ("PASS" if ok else "FAIL", name, "" if ok else "  -> " + why))
    n_fail = sum(1 for _, ok, _ in results if not ok)
    print("%d/%d passed" % (len(results) - n_fail, len(results)))
    return 1 if n_fail else 0


if __name__ == "__main__":
    if "--selftest" in sys.argv[1:]:
        sys.exit(_selftest())
    print(__doc__)
    sys.exit(0)
