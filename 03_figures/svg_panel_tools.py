#!/usr/bin/env python3
"""Shared figure settings

Schematic SVG helpers: rasterise a hand-authored panel SVG to PNG and derive a PowerPoint-friendly copy

Imported by the figure scripts; it renders no panel of its own.
"""

from __future__ import annotations
import argparse
import math
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image

EDGE = [Path("C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe"),
        Path("C:/Program Files/Microsoft/Edge/Application/msedge.exe"),
        Path("/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge"),
        Path("/Applications/Google Chrome.app/Contents/MacOS/Google Chrome")]


def viewbox(svg_text: str) -> tuple[float, float]:
    m = re.search(r'viewBox="([\d.\s-]+)"', svg_text)
    _, _, w, h = (float(v) for v in m.group(1).split())
    return w, h


def rasterise(svg: Path, scale: int = 3, pad: int = 16) -> Path:
    text = svg.read_text(encoding="utf-8")
    w, h = viewbox(text)
    out = svg.with_suffix(".png")
    tmp = svg.resolve().with_name("_tmp_" + svg.stem + "_raster.png")
    edge = next((p for p in EDGE if p.exists()), None)
    if edge is None:
        sys.exit(f"msedge.exe not found; on macOS: qlmanage -t -s {int(w*scale)} -o <dir> {svg}")
    subprocess.run([str(edge), "--headless=new", "--disable-gpu", "--hide-scrollbars",
                    f"--force-device-scale-factor={scale}", f"--window-size={int(w)},{int(h)}",
                    f"--screenshot={tmp}", svg.resolve().as_uri()], check=True,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    im = Image.open(tmp).convert("RGB")
    a = np.asarray(im)
    ys, xs = np.where((a < 250).any(axis=2))
    box = (max(xs.min() - pad, 0), max(ys.min() - pad, 0),
           min(xs.max() + pad, a.shape[1]), min(ys.max() + pad, a.shape[0]))
    im.crop(box).save(out, dpi=(300, 300))
    tmp.unlink(missing_ok=True)
    print(f"png  {out}  {im.crop(box).size}")
    return out


def ppt_variant(svg: Path) -> Path:
    s = svg.read_text(encoding="utf-8")
    style = re.search(r"<style>(.*?)</style>", s, flags=re.S)
    if style:
        classes = {}
        for name, body in re.findall(r"\.([\w-]+)\s*\{([^}]*)\}", style.group(1)):
            kv_pairs = [(k.strip(), v.strip()) for k, v in (kv.split(":", 1) for kv in body.split(";") if ":" in kv)]
            halo = any(k == "paint-order" for k, _ in kv_pairs)
            skip = {"paint-order"} | ({"stroke", "stroke-width", "stroke-linejoin"} if halo else set())
            classes[name] = [(k, v) for k, v in kv_pairs if k not in skip]

        def inline(m):
            tag = m.group(0)
            name = re.search(r'class="([^"]+)"', tag).group(1)
            present = set(re.findall(r'\s([\w:-]+)="', tag))
            attrs = " ".join(f'{k}="{v}"' for k, v in classes.get(name, []) if k not in present)
            return tag.replace(f'class="{name}"', attrs, 1)

        s = re.sub(r"<[^>]*\bclass=\"[^\"]+\"[^>]*>", inline, s)
        s = re.sub(r"\s*<style>.*?</style>", "", s, flags=re.S)
    s = re.sub(r'\s*paint-order="stroke"\s*stroke="#ffffff"\s*stroke-width="\d+(?:px)?"', "", s)
    s = re.sub(r'\s*paint-order="stroke"', "", s)
    colours = {m.group(1): m.group(2) for m in re.finditer(
        r'<marker id="([^"]+)".*?fill="(#[0-9a-fA-F]{6})"', s, flags=re.S)}
    tris = []

    def line_sub(m):
        tag = m.group(0)
        mk = re.search(r'marker-end="url\(#([^)]+)\)"', tag).group(1)
        g = lambda k: float(re.search(rf'\b{k}="([-\d.]+)"', tag).group(1))
        x1, y1, x2, y2 = g("x1"), g("y1"), g("x2"), g("y2")
        sw = re.search(r'stroke-width="([\d.]+)"', tag)
        L = min(4.0 * (float(sw.group(1)) if sw else 1.6) + 4, 14)
        ang = math.atan2(y2 - y1, x2 - x1)
        bx, by = x2 - L * math.cos(ang), y2 - L * math.sin(ang)
        nx, ny = -math.sin(ang) * L * 0.5, math.cos(ang) * L * 0.5
        tri = (f'<polygon points="{x2:.1f},{y2:.1f} {bx+nx:.1f},{by+ny:.1f} {bx-nx:.1f},{by-ny:.1f}" '
               f'fill="{colours.get(mk, "#555555")}" stroke="none"/>')
        tris.append(tri)
        return re.sub(r'\s*marker-end="[^"]+"', "", tag) + tri

    s = re.sub(r"<line\b[^>]*marker-end=[^>]*/>", line_sub, s)

    def path_sub(m):
        tag = m.group(0)
        mk = re.search(r'marker-end="url\(#([^)]+)\)"', tag).group(1)
        d = re.search(r'\bd="([^"]+)"', tag).group(1)
        nums = [float(v) for v in re.findall(r"-?\d+(?:\.\d+)?", d)]
        if "C" in d or "Q" in d or "A" in d or len(nums) < 4:
            print("WARNING: curved <path> with marker-end needs a hand-placed arrowhead:", tag[:80])
            return tag
        x1, y1, x2, y2 = nums[-4], nums[-3], nums[-2], nums[-1]
        sw = re.search(r'stroke-width="([\d.]+)"', tag)
        L = min(4.0 * (float(sw.group(1)) if sw else 1.6) + 4, 14)
        ang = math.atan2(y2 - y1, x2 - x1)
        bx, by = x2 - L * math.cos(ang), y2 - L * math.sin(ang)
        nx, ny = -math.sin(ang) * L * 0.5, math.cos(ang) * L * 0.5
        tri = (f'<polygon points="{x2:.1f},{y2:.1f} {bx+nx:.1f},{by+ny:.1f} {bx-nx:.1f},{by-ny:.1f}" '
               f'fill="{colours.get(mk, "#555555")}" stroke="none"/>')
        tris.append(tri)
        return re.sub(r'\s*marker-end="[^"]+"', "", tag) + tri

    s = re.sub(r"<path\b[^>]*marker-end=[^>]*/>", path_sub, s)

    def start_sub(m):
        tag = m.group(0)
        mk = re.search(r'marker-start="url\(#([^)]+)\)"', tag).group(1)
        g = lambda k: float(re.search(rf'\b{k}="([-\d.]+)"', tag).group(1))
        x1, y1, x2, y2 = g("x1"), g("y1"), g("x2"), g("y2")
        L = 10.0
        ang = math.atan2(y1 - y2, x1 - x2)
        bx, by = x1 - L * math.cos(ang), y1 - L * math.sin(ang)
        nx, ny = -math.sin(ang) * L * 0.5, math.cos(ang) * L * 0.5
        tri = (f'<polygon points="{x1:.1f},{y1:.1f} {bx+nx:.1f},{by+ny:.1f} {bx-nx:.1f},{by-ny:.1f}" '
               f'fill="{colours.get(mk, "#555555")}" stroke="none"/>')
        tris.append(tri)
        return re.sub(r'\s*marker-start="[^"]+"', "", tag) + tri

    s = re.sub(r"<line\b[^>]*marker-start=[^>]*/>", start_sub, s)
    s = re.sub(r"\s*<defs>.*?</defs>", "", s, flags=re.S) if "<clipPath" not in s and "Gradient" not in s \
        else re.sub(r"\s*<marker\b.*?</marker>", "", s, flags=re.S)
    s = s.replace("</svg>", f"  <!-- {len(tris)} explicit arrowheads inserted beside their lines "
                            "(PowerPoint drops SVG markers) -->\n</svg>")
    s = s.replace("<!--", "<!-- PPT-FRIENDLY VARIANT derived by 03_figures/svg_panel_tools.py; edit the master SVG. ", 1)
    out = svg.with_name(svg.stem + "_ppt.svg")
    out.write_text(s, encoding="utf-8")
    print(f"ppt  {out}  (classes left: {len(re.findall(r'class=', s))}, markers left: {len(re.findall(r'marker-end', s))})")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("svg", nargs="+")
    ap.add_argument("--scale", type=int, default=3)
    ap.add_argument("--no-ppt", action="store_true")
    a = ap.parse_args()
    for p in a.svg:
        p = Path(p)
        rasterise(p, a.scale)
        if not a.no_ppt:
            ppt_variant(p)
