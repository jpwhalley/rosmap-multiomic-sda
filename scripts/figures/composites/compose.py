#!/usr/bin/env python
"""compose.py: build a composite figure from panel files with xelatex + TikZ, then check it.

Usage
  compose.py LAYOUT.json [--workdir DIR] [--name NAME] [--final DEST.pdf] [--dpi 150] [--min-pt 5]
                         [--debug-boxes]
  compose.py --check PDF [--min-pt 5] [--placed-width-mm 142.9]
                                               # only the text-size and font checks, on any PDF (sizes as printed
                                               # at the given width, e.g. an Illustrator export at \textwidth)
  compose.py --panel-info FILE [FILE ...]      # native size, fonts, smallest text, width needed for 5/6 pt,
                                               # ink bounding box and a suggested trim_mm for white margins

Layout JSON (all lengths in mm, origin = top-left of the composite, y downwards):
  {
    "width_mm": 142.9, "height_mm": 120,
    "letter_size_pt": 8,                       # optional (default 8; Arial Bold)
    "background_white": true,                  # optional (default true): white rectangle under everything
    "cells": [
      {"file": "$G_ROS_FIGURE_OUT/S1/panels/panel.pdf",  # PDF (vector), PNG or JPG; $VARS and ~ are expanded
       "x_mm": 0, "y_mm": 0,                   # top-left corner of the placed panel
       "w_mm": 70,                             # one of w_mm / h_mm / scale (aspect ratio always kept)
       "page": 1,                              # optional, for multi-page PDFs
       "trim_mm": [0, 0, 0, 0],                # optional [left, bottom, right, top] crop of the native file
       "letter": "a",                          # optional panel letter
       "letter_dx_mm": 0, "letter_dy_mm": 0    # letter top-left = (x_mm + dx, y_mm + dy); default 0, 0
      }],
    "labels": [{"letter": "d", "x_mm": 0, "y_mm": 60}],     # optional free letters (one letter, several files)
    "extra_text": [{"text": "Male", "x_mm": 10, "y_mm": 5, "size_pt": 7,
                    "bold": false, "italic": false, "anchor": "north west", "rotate": 0, "raw": false}]
  }
Panel letters: Arial Bold, letter_size_pt, the top of an ascender (b, d, f ...) at the given y, so letters with
the same y share a baseline. Each letter is drawn last (on top of panels).

Outputs in the workdir (default: <layout dir>/build_<name>/): <name>.tex, <name>.pdf, <name>.png (preview at
--dpi), <name>_report.json, inputs/ (copies of the panel files). --final copies the PDF to DEST only if DEST does
not exist (never overwrites). --debug-boxes also writes <name>_boxes.png with every cell outlined (not for use).

Checks (printed and in the report):
  page size     pdfinfo: exact width x height (tolerance 0.02 mm)
  text size     (1) ghostscript txtwrite: exact effective font size of every text span (scaling included), the
                    primary measure (calibration: within 0.002 pt of known sizes; rendered cap heights agree to
                    within 3%); (2) pdftotext -bbox: word box heights, converted to font size with the matplotlib-Arial
                    factor 1.118 (orientation from the matched span). The box/size factor depends on the
                    embedding program (matplotlib Arial 1.118, xelatex Arial 0.938, Illustrator Arial 1.331,
                    Illustrator Palatino-Bold 1.658), so (2) alone is off by -16% to +19% (+48%): the report lists the
                    factor per font, and a word is flagged by its exact size whenever gs sees it.
                Text that is outlined (paths) or inside raster images is invisible to both: such cells are listed.
  Print size    the supplement includes composites at width=\textwidth = 142.9 mm: build at width_mm 142.9 so
                that the checked sizes are the printed sizes (otherwise use --check --placed-width-mm).
  fonts         pdffonts on the output and on every input: Type 3 = FAIL, not embedded = FAIL, non-Arial = listed
                with the cells that bring it (text inside placed PDFs keeps its own fonts).
  layout        cells outside the page or overlapping; letters near-but-not-exactly aligned in rows/columns.
"""
import argparse, html, json, os, re, shutil, subprocess, sys
from pathlib import Path

MM_PER_BP = 25.4 / 72.0          # PDF point (bp)
TEXPT_PER_BP = 72.27 / 72.0      # TeX pt per PostScript point: letter/text sizes are given in PostScript points
ARIAL_K = 1.118                  # pdftotext word-box height / font size for Arial (calibrated on panels of known size)
TOL_MM = 0.02


def run(cmd, cwd=None, check=True):
    p = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, errors="replace")
    if check and p.returncode != 0:
        raise RuntimeError(f"command failed ({p.returncode}): {' '.join(map(str, cmd))}\n{p.stdout[-3000:]}\n{p.stderr[-3000:]}")
    return p


# ----------------------------------------------------------------------------------------------- inspection
def pdf_box_bp(path, page=1):
    """CropBox (falls back to MediaBox) of one page, in bp: (x0, y0, x1, y1)."""
    out = run(["pdfinfo", "-box", "-f", str(page), "-l", str(page), str(path)]).stdout
    boxes = {}
    for m in re.finditer(r"Page\s+\d+\s+(MediaBox|CropBox):\s+([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)", out):
        boxes[m.group(1)] = tuple(float(m.group(i)) for i in range(2, 6))
    if not boxes:  # single-page output format
        for m in re.finditer(r"^(MediaBox|CropBox):\s+([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)", out, re.M):
            boxes[m.group(1)] = tuple(float(m.group(i)) for i in range(2, 6))
    b = boxes.get("CropBox") or boxes.get("MediaBox")
    if b is None:
        raise RuntimeError(f"no page box for {path}: {out[:500]}")
    rot = re.search(r"Page\s+\d+\s+rot:\s+(\d+)", out) or re.search(r"^Page rot:\s+(\d+)", out, re.M)
    if rot and int(rot.group(1)) % 180:
        raise RuntimeError(f"{path} page {page} is rotated {rot.group(1)} deg; not supported")
    return b


def pdf_npages(path):
    m = re.search(r"^Pages:\s+(\d+)", run(["pdfinfo", str(path)]).stdout, re.M)
    return int(m.group(1)) if m else 1


def pdffonts(path, page=None):
    cmd = ["pdffonts"] + (["-f", str(page), "-l", str(page)] if page else []) + [str(path)]
    lines = run(cmd).stdout.splitlines()
    fonts = []
    if len(lines) < 2:
        return fonts
    hdr, rule = lines[0], lines[1]
    # column starts from the dashed rule
    starts = [m.start() for m in re.finditer(r"-+", rule)]
    ends = [m.end() for m in re.finditer(r"-+", rule)]
    for ln in lines[2:]:
        cols = [ln[s:e if i < len(starts) - 1 else None].strip() for i, (s, e) in enumerate(zip(starts, ends))]
        if len(cols) < 6:
            continue
        name, ftype, enc, emb, sub, uni = cols[:6]
        fonts.append(dict(name=name, type=ftype, encoding=enc, emb=emb, sub=sub, uni=uni))
    return fonts


def font_family(name):
    return re.sub(r"^[A-Z]{6}\+", "", name or "")


def is_arial(name):
    n = font_family(name).replace(" ", "").lower()
    return n.startswith("arial") and "narrow" not in n and "rounded" not in n and "unicode" not in n


def gs_spans(pdf, page=None):
    """Exact effective font size of every text span (ghostscript txtwrite). Coordinates in bp from top-left."""
    cmd = ["gs", "-q", "-dNOPAUSE", "-dBATCH", "-dSAFER", "-sDEVICE=txtwrite", "-dTextFormat=0"]
    if page:
        cmd += [f"-dFirstPage={page}", f"-dLastPage={page}"]
    cmd += ["-sOutputFile=-", str(pdf)]
    out = run(cmd, check=False).stdout
    spans = []
    for m in re.finditer(r'<span bbox="([-\d.]+) ([-\d.]+) ([-\d.]+) ([-\d.]+)" font="([^"]*)" size="([\d.]+)">(.*?)</span>',
                         out, re.S):
        x0, y0, x1, y1 = (float(m.group(i)) for i in range(1, 5))
        chars = "".join(html.unescape(c) for c in re.findall(r' c="([^"]*)"', m.group(7)))
        if not chars.strip():
            continue
        if abs(y0 - y1) < 0.5:
            orient = "h"
        elif abs(x0 - x1) < 0.5:
            orient = "v"
        else:
            orient = "diag"
        spans.append(dict(text=chars.strip(), size=float(m.group(6)), font=m.group(5), orient=orient,
                          x0=x0, y0=y0, x1=x1, y1=y1))
    return spans


def bbox_words(pdf, page=None):
    """pdftotext -bbox words: dicts with text and xMin, yMin, xMax, yMax in bp from top-left."""
    cmd = ["pdftotext", "-bbox"] + (["-f", str(page), "-l", str(page)] if page else []) + [str(pdf), "-"]
    out = run(cmd).stdout
    words = []
    for m in re.finditer(r'<word xMin="([-\d.]+)" yMin="([-\d.]+)" xMax="([-\d.]+)" yMax="([-\d.]+)">(.*?)</word>', out):
        words.append(dict(text=html.unescape(m.group(5)), xMin=float(m.group(1)), yMin=float(m.group(2)),
                          xMax=float(m.group(3)), yMax=float(m.group(4))))
    return words


def match_word_to_spans(w, spans, tol=1.6):
    """Spans (gs) that geometrically hold this pdftotext word. gs rounds coordinates to whole bp."""
    hits = []
    for s in spans:
        if s["orient"] == "h":
            if w["yMin"] - tol <= s["y0"] <= w["yMax"] + tol:
                lo, hi = min(s["x0"], s["x1"]), max(s["x0"], s["x1"])
                if lo - tol <= w["xMax"] and hi + tol >= w["xMin"]:
                    hits.append(s)
        elif s["orient"] == "v":
            if w["xMin"] - tol <= s["x0"] <= w["xMax"] + tol:
                lo, hi = min(s["y0"], s["y1"]), max(s["y0"], s["y1"])
                if lo - tol <= w["yMax"] and hi + tol >= w["yMin"]:
                    hits.append(s)
    if len(hits) > 1:  # prefer spans whose text contains the word
        t = w["text"].replace(" ", "")
        texty = [s for s in hits if t and t in s["text"].replace(" ", "")]
        if texty:
            hits = texty
    return hits


def text_size_check(pdf, min_pt=5.0, cells=None, page=None, scale=1.0):
    """Both measures; cells (optional) = list of dicts with id/file/bbox_mm used to attribute flagged text.
    scale (optional) = placement scale of this PDF (e.g. --placed-width-mm / native width): sizes and positions
    are reported as printed."""
    spans = gs_spans(pdf, page)
    words = bbox_words(pdf, page)
    if scale != 1.0:
        for s_ in spans:
            s_["size"] *= scale
            for k in ("x0", "y0", "x1", "y1"):
                s_[k] *= scale
        for w_ in words:
            for k in ("xMin", "yMin", "xMax", "yMax"):
                w_[k] *= scale

    def where(x_mm, y_mm):
        if not cells:
            return None
        for c in cells:
            x0, y0, x1, y1 = c["bbox_mm"]
            if x0 - 0.3 <= x_mm <= x1 + 0.3 and y0 - 0.3 <= y_mm <= y1 + 0.3:
                return f"cell {c['id']} ({Path(c['file']).name}{', ' + c['letter'] if c.get('letter') else ''})"
        return "outside cells (letters/extra text)"

    gs_rows = []
    for s in spans:
        x_mm, y_mm = min(s["x0"], s["x1"]) * MM_PER_BP, min(s["y0"], s["y1"]) * MM_PER_BP
        gs_rows.append(dict(text=s["text"][:60], size_pt=round(s["size"], 2), orient=s["orient"],
                            font=font_family(s["font"]).replace("-Identity-H", ""), x_mm=round(x_mm, 1),
                            y_mm=round(y_mm, 1), where=where(x_mm, y_mm)))
    w_rows, unmatched, ratios = [], 0, {}
    for w in words:
        hits = match_word_to_spans(w, spans)
        orient = hits[0]["orient"] if hits else "h?"
        h, wd = w["yMax"] - w["yMin"], w["xMax"] - w["xMin"]
        size_dim = wd if orient == "v" else h
        gs_size = min(s["size"] for s in hits) if hits else None
        if not hits:
            unmatched += 1
        elif len({round(s["size"], 3) for s in hits}) == 1 and gs_size > 0:
            fam = font_family(hits[0]["font"]).replace("-Identity-H", "")
            ratios.setdefault(fam, []).append(size_dim / gs_size)
        x_mm, y_mm = w["xMin"] * MM_PER_BP, w["yMin"] * MM_PER_BP
        w_rows.append(dict(text=w["text"][:60], box_h_pt=round(h, 2), box_w_pt=round(wd, 2), orient=orient,
                           est_size_pt=round(size_dim / ARIAL_K, 2),
                           gs_size_pt=None if gs_size is None else round(gs_size, 2),
                           x_mm=round(x_mm, 1), y_mm=round(y_mm, 1), where=where(x_mm, y_mm)))
    res = dict(n_gs_spans=len(gs_rows), n_bbox_words=len(w_rows), n_words_unmatched=unmatched)
    if gs_rows:
        res["gs_min_size_pt"] = min(r["size_pt"] for r in gs_rows)
        res["gs_below_min"] = sorted([r for r in gs_rows if r["size_pt"] < min_pt - 1e-6], key=lambda r: r["size_pt"])
    if w_rows:
        hor = [r for r in w_rows if r["orient"] != "v"]
        res["bbox_min_word_height_pt"] = min(r["box_h_pt"] for r in hor) if hor else None
        res["bbox_min_est_size_pt"] = min(r["est_size_pt"] for r in w_rows)
        # a word is flagged by the exact size when it has one, else by the box estimate
        res["bbox_below_min"] = sorted([r for r in w_rows if (r["gs_size_pt"] is not None and r["gs_size_pt"] < min_pt - 1e-6)
                                        or (r["gs_size_pt"] is None and r["est_size_pt"] < min_pt - 0.05)],
                                       key=lambda r: r["est_size_pt"])
        res["bbox_estimate_false_alarms"] = sum(1 for r in w_rows if r["est_size_pt"] < min_pt - 0.05
                                                and r["gs_size_pt"] is not None and r["gs_size_pt"] >= min_pt)
        res["bbox_box_over_size_by_font"] = {f: [round(min(v), 3), round(sorted(v)[len(v) // 2], 3), round(max(v), 3), len(v)]
                                             for f, v in ratios.items()}   # [min, median, max, n]; Arial via matplotlib = 1.118
    res["min_pt"] = min_pt
    return res


def font_check(pdf, cell_fonts=None):
    fonts = pdffonts(pdf)
    rows, fails, notes = [], [], []
    for f in fonts:
        src = []
        if cell_fonts:
            src = [cid for cid, fl in cell_fonts.items() if any(font_family(g["name"]) == font_family(f["name"]) for g in fl)]
        r = dict(f, family=font_family(f["name"]), arial=is_arial(f["name"]), from_cells=src)
        rows.append(r)
        if "type 3" in f["type"].lower():
            fails.append(f"Type 3 font {r['family']} (cells {src or '?'})")
        if f["emb"] != "yes":
            fails.append(f"font not embedded: {r['family']} (cells {src or '?'})")
        if not r["arial"]:
            notes.append(f"non-Arial font {r['family']} [{f['type']}] (cells {src or 'composite text'})")
    return dict(fonts=rows, fails=fails, non_arial=notes)


def ink_bbox_mm(pdf, page=1, dpi=150, thresh=245):
    """Bounding box of non-white pixels (x0, y0, x1, y1 in mm from the top-left), from a pdftoppm render."""
    import tempfile
    import numpy as np
    from PIL import Image
    with tempfile.TemporaryDirectory() as td:
        base = Path(td) / "r"
        run(["pdftoppm", "-r", str(dpi), "-f", str(page), "-l", str(page), "-png", "-singlefile", str(pdf), str(base)])
        a = np.asarray(Image.open(str(base) + ".png").convert("L"))
    ys, xs = np.nonzero(a < thresh)
    if len(xs) == 0:
        return None
    k = 25.4 / dpi
    return (xs.min() * k, ys.min() * k, (xs.max() + 1) * k, (ys.max() + 1) * k)


def panel_info(path, page=1):
    path = Path(path)
    ext = path.suffix.lower()
    info = dict(file=str(path))
    if ext == ".pdf":
        x0, y0, x1, y1 = pdf_box_bp(path, page)
        info.update(kind="pdf", page=page, n_pages=pdf_npages(path), w_mm=round((x1 - x0) * MM_PER_BP, 2),
                    h_mm=round((y1 - y0) * MM_PER_BP, 2))
        sp = gs_spans(path, page)
        info["n_text_spans"] = len(sp)
        if sp:
            sizes = sorted(s["size"] for s in sp)
            info["min_text_pt"] = round(sizes[0], 2)
            info["median_text_pt"] = round(sizes[len(sizes) // 2], 2)
            info["min_text_examples"] = [s["text"][:30] for s in sp if s["size"] <= sizes[0] + 1e-6][:5]
            info["min_width_mm_for_5pt"] = round(info["w_mm"] * 5.0 / sizes[0], 1)
            info["min_width_mm_for_6pt"] = round(info["w_mm"] * 6.0 / sizes[0], 1)
        ib = ink_bbox_mm(path, page)
        if ib:
            info["ink_bbox_mm"] = [round(v, 1) for v in ib]          # x0, y0, x1, y1 from the top-left
            pad = 0.5
            info["suggested_trim_mm"] = [round(max(0.0, ib[0] - pad), 1), round(max(0.0, info["h_mm"] - ib[3] - pad), 1),
                                         round(max(0.0, info["w_mm"] - ib[2] - pad), 1), round(max(0.0, ib[1] - pad), 1)]
        fl = pdffonts(path, page)
        info["fonts"] = [f"{font_family(f['name'])} [{f['type']}{'' if f['emb'] == 'yes' else ', NOT embedded'}]" for f in fl]
        info["type3"] = any("type 3" in f["type"].lower() for f in fl)
    else:
        from PIL import Image
        with Image.open(path) as im:
            info.update(kind="raster", px=im.size, dpi=im.info.get("dpi"))
    return info


# --------------------------------------------------------------------------------------------- composition
def tex_escape(s):
    rep = {"\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$", "#": r"\#", "_": r"\_",
           "{": r"\{", "}": r"\}", "~": r"\textasciitilde{}", "^": r"\textasciicircum{}"}
    return "".join(rep.get(ch, ch) for ch in s)


def f3(v):
    return f"{v:.3f}"


def prepare_cells(layout, inputs_dir):
    cells = []
    for i, c in enumerate(layout["cells"], 1):
        src = Path(os.path.expandvars(c["file"])).expanduser()
        if not src.exists():
            raise FileNotFoundError(f"cell {i}: {src}")
        ext = src.suffix.lower()
        if ext not in (".pdf", ".png", ".jpg", ".jpeg"):
            raise ValueError(f"cell {i}: unsupported file type {ext}")
        dst = inputs_dir / f"c{i:02d}{ext}"
        shutil.copyfile(src, dst)                        # read-only on the source
        page = int(c.get("page", 1))
        trim = [float(v) for v in c.get("trim_mm", [0, 0, 0, 0])]
        if ext == ".pdf":
            x0, y0, x1, y1 = pdf_box_bp(dst, page)
            nat_w, nat_h = (x1 - x0) * MM_PER_BP, (y1 - y0) * MM_PER_BP
            px = None
        else:
            from PIL import Image
            with Image.open(dst) as im:
                px = im.size
                dpi = im.info.get("dpi", (72, 72))[0] or 72
            # graphicx sizes rasters from their dpi; only the aspect ratio matters because a size is always given
            nat_w, nat_h = px[0] / dpi * 25.4, px[1] / dpi * 25.4
        eff_w, eff_h = nat_w - trim[0] - trim[2], nat_h - trim[1] - trim[3]
        if eff_w <= 0 or eff_h <= 0:
            raise ValueError(f"cell {i}: trim removes the whole panel")
        n_given = sum(k in c for k in ("w_mm", "h_mm", "scale"))
        if n_given != 1:
            raise ValueError(f"cell {i}: give exactly one of w_mm, h_mm, scale")
        if "w_mm" in c:
            w = float(c["w_mm"]); h = w * eff_h / eff_w
        elif "h_mm" in c:
            h = float(c["h_mm"]); w = h * eff_w / eff_h
        else:
            w = eff_w * float(c["scale"]); h = eff_h * float(c["scale"])
        x, y = float(c["x_mm"]), float(c["y_mm"])
        cell = dict(id=i, file=str(src), local=dst.name, ext=ext, page=page, trim_mm=trim, x=x, y=y, w=w, h=h,
                    bbox_mm=(x, y, x + w, y + h), native_w_mm=round(eff_w, 2), native_h_mm=round(eff_h, 2),
                    scale=round(w / eff_w, 4), letter=c.get("letter"),
                    ldx=float(c.get("letter_dx_mm", 0.0)), ldy=float(c.get("letter_dy_mm", 0.0)))
        if px:
            cell["raster_px"] = px
            cell["effective_dpi"] = round(px[0] / (w / 25.4 * (nat_w / eff_w)), 0) if eff_w else None
        cells.append(cell)
    return cells


def write_tex(layout, cells, tex_path, debug_boxes=False):
    W, H = float(layout["width_mm"]), float(layout["height_mm"])
    lsize = float(layout.get("letter_size_pt", 8)) * TEXPT_PER_BP     # sizes are PostScript points (as measured)
    L = [r"\documentclass[tikz,border=0pt]{standalone}",
         r"\usepackage{fontspec}",
         r"\setmainfont{Arial}[BoldFont={Arial Bold},ItalicFont={Arial Italic},BoldItalicFont={Arial Bold Italic}]",
         r"\usepackage{graphicx}",
         r"\begin{document}",
         r"\begin{tikzpicture}[x=1mm,y=-1mm]",
         rf"\useasboundingbox (0,0) rectangle ({f3(W)},{f3(H)});"]
    if layout.get("background_white", True):
        L.append(rf"\fill[white] (0,0) rectangle ({f3(W)},{f3(H)});")
    for c in cells:
        opts = [rf"width={f3(c['w'])}mm"]
        if c["ext"] == ".pdf" and c["page"] != 1:
            opts.append(f"page={c['page']}")
        if any(c["trim_mm"]):
            opts.append("trim=" + " ".join(f"{v / MM_PER_BP:.3f}bp" for v in c["trim_mm"]) + ",clip")
        L.append(rf"\node[anchor=north west,inner sep=0pt,outer sep=0pt] at ({f3(c['x'])},{f3(c['y'])}) "
                 rf"{{\includegraphics[{','.join(opts)}]{{inputs/{c['local']}}}}}; % cell {c['id']}: {tex_escape(Path(c['file']).name)}")
    for t in layout.get("extra_text", []):
        size = float(t.get("size_pt", 7)) * TEXPT_PER_BP
        txt = t["text"] if t.get("raw") else tex_escape(t["text"])
        if t.get("bold"):
            txt = r"\bfseries " + txt
        if t.get("italic"):
            txt = r"\itshape " + txt
        L.append(rf"\node[anchor={t.get('anchor', 'north west')},inner sep=0pt,outer sep=0pt,rotate={float(t.get('rotate', 0))},"
                 rf"font=\fontsize{{{size:.3f}}}{{{size * 1.2:.3f}}}\selectfont] at ({f3(float(t['x_mm']))},{f3(float(t['y_mm']))}) {{{txt}}};")
    letters = letter_positions(layout, cells)
    for lt in letters:
        L.append(rf"\node[anchor=north west,inner sep=0pt,outer sep=0pt,font=\fontsize{{{lsize:.3f}}}{{{lsize * 1.2:.3f}}}\selectfont\bfseries] "
                 rf"at ({f3(lt['x'])},{f3(lt['y'])}) {{\vphantom{{bdfhkl}}{tex_escape(lt['letter'])}}};")
    if debug_boxes:
        for c in cells:
            L.append(rf"\draw[red,line width=0.2pt] ({f3(c['x'])},{f3(c['y'])}) rectangle ({f3(c['x'] + c['w'])},{f3(c['y'] + c['h'])});")
            L.append(rf"\node[red,anchor=south east,inner sep=0.5pt,font=\fontsize{{6}}{{7}}\selectfont] at ({f3(c['x'] + c['w'])},{f3(c['y'] + c['h'])}) {{{c['id']}}};")
        L.append(rf"\draw[blue,line width=0.2pt] (0,0) rectangle ({f3(W)},{f3(H)});")
    L += [r"\end{tikzpicture}", r"\end{document}", ""]
    tex_path.write_text("\n".join(L))
    return letters


def letter_positions(layout, cells):
    out = []
    for c in cells:
        if c.get("letter"):
            out.append(dict(letter=c["letter"], x=c["x"] + c["ldx"], y=c["y"] + c["ldy"], src=f"cell {c['id']}"))
    for lab in layout.get("labels", []):
        out.append(dict(letter=lab["letter"], x=float(lab["x_mm"]), y=float(lab["y_mm"]), src="labels"))
    return out


def layout_checks(layout, cells, letters):
    W, H = float(layout["width_mm"]), float(layout["height_mm"])
    warn = []
    for c in cells:
        x0, y0, x1, y1 = c["bbox_mm"]
        if x0 < -TOL_MM or y0 < -TOL_MM or x1 > W + TOL_MM or y1 > H + TOL_MM:
            warn.append(f"cell {c['id']} ({Path(c['file']).name}) extends outside the page: "
                        f"[{x0:.2f}, {y0:.2f}, {x1:.2f}, {y1:.2f}] vs {W} x {H}")
        if c.get("effective_dpi") and c["effective_dpi"] < 300:
            warn.append(f"cell {c['id']} raster at {c['effective_dpi']:.0f} dpi effective (< 300)")
    for i, a in enumerate(cells):
        for b in cells[i + 1:]:
            ox = min(a["bbox_mm"][2], b["bbox_mm"][2]) - max(a["bbox_mm"][0], b["bbox_mm"][0])
            oy = min(a["bbox_mm"][3], b["bbox_mm"][3]) - max(a["bbox_mm"][1], b["bbox_mm"][1])
            if ox > 0.05 and oy > 0.05:
                warn.append(f"cells {a['id']} and {b['id']} overlap by {ox:.1f} x {oy:.1f} mm "
                            "(a white panel background may hide the other panel)")
    for lt in letters:
        if not re.fullmatch(r"[a-z]", lt["letter"]):
            warn.append(f"letter '{lt['letter']}' ({lt['src']}) is not a single lower-case letter")
        if not (-TOL_MM <= lt["x"] <= W and -TOL_MM <= lt["y"] <= H):
            warn.append(f"letter '{lt['letter']}' outside the page")
    for key, name in (("y", "row"), ("x", "column")):
        vals = sorted(lt[key] for lt in letters)
        for a, b in zip(vals, vals[1:]):
            if 0.01 < b - a < 2.0:
                who = [lt["letter"] for lt in letters if abs(lt[key] - a) < 0.01 or abs(lt[key] - b) < 0.01]
                warn.append(f"letters {who} nearly share a {name} ({key} = {a:.2f} vs {b:.2f} mm); align them exactly?")
    seen = {}
    for lt in letters:
        seen.setdefault(lt["letter"], []).append(lt["src"])
    for k, v in seen.items():
        if len(v) > 1:
            warn.append(f"letter '{k}' placed {len(v)} times ({v})")
    return warn


def compile_tex(tex_path):
    p = run(["xelatex", "-interaction=nonstopmode", "-halt-on-error", tex_path.name], cwd=tex_path.parent, check=False)
    log = tex_path.with_suffix(".log").read_text(errors="replace") if tex_path.with_suffix(".log").exists() else ""
    if p.returncode != 0 or not tex_path.with_suffix(".pdf").exists():
        tail = "\n".join(log.splitlines()[-40:])
        raise RuntimeError(f"xelatex failed for {tex_path}:\n{tail}")
    missing = sorted(set(re.findall(r"Missing character: There is no (.+?) in font", log)))
    return dict(missing_glyphs=missing,
                warnings=sorted(set(re.findall(r"(LaTeX Warning: .*|Package \w+ Warning: .*)", log)))[:20])


def page_size_mm(pdf):
    out = run(["pdfinfo", "-box", str(pdf)]).stdout
    m = re.search(r"MediaBox:\s+([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)", out)
    x0, y0, x1, y1 = (float(m.group(i)) for i in range(1, 5))
    return (x1 - x0) * MM_PER_BP, (y1 - y0) * MM_PER_BP


def render_png(pdf, png_base, dpi):
    run(["pdftoppm", "-r", str(dpi), "-png", "-singlefile", str(pdf), str(png_base)])
    return Path(str(png_base) + ".png")


def summarise(rep, out=sys.stdout):
    p = lambda *a: print(*a, file=out)
    p(f"== {rep.get('name', rep.get('pdf'))}")
    if "pdf" in rep:
        p(f"   pdf      {rep['pdf']}")
    if "png" in rep:
        p(f"   preview  {rep['png']}")
    if "page_size_mm" in rep:
        ok = "OK" if rep["page_size_ok"] else "MISMATCH"
        p(f"   page     {rep['page_size_mm'][0]:.2f} x {rep['page_size_mm'][1]:.2f} mm "
          f"(requested {rep['requested_mm'][0]} x {rep['requested_mm'][1]}) {ok}")
    for c in rep.get("cells", []):
        extra = f", raster {c['raster_px'][0]}x{c['raster_px'][1]} px = {c['effective_dpi']:.0f} dpi" if c.get("raster_px") else ""
        mt = c.get("native_min_text_pt")
        tx = f", text {mt} pt native -> {c['placed_min_text_pt']} pt placed" if mt else (", NO extractable text" if c["ext"] == ".pdf" else "")
        p(f"   cell {c['id']:>2} {c.get('letter') or '-'}  {Path(c['file']).name}: {c['w']:.1f} x {c['h']:.1f} mm at "
          f"({c['x']:.1f}, {c['y']:.1f}), scale {c['scale']}{tx}{extra}")
    ts = rep["text_size"]
    if ts.get("n_gs_spans"):
        p(f"   text     smallest {ts['gs_min_size_pt']} pt (ghostscript, exact); smallest word box height "
          f"{ts.get('bbox_min_word_height_pt')} pt = about {ts.get('bbox_min_est_size_pt')} pt type (pdftotext -bbox, / {ARIAL_K}); "
          f"{ts['n_words_unmatched']} of {ts['n_bbox_words']} words unmatched")
    else:
        p("   text     no extractable text")
    if ts.get("bbox_box_over_size_by_font"):
        p("   bbox     word box / exact size by font: " + "; ".join(
            f"{f} {v[1]} (n {v[3]})" for f, v in ts["bbox_box_over_size_by_font"].items()))
    below = ts.get("gs_below_min", [])
    p(f"   < {ts['min_pt']} pt: {len(below)} span(s)" + (":" if below else ""))
    for r in below[:40]:
        p(f"      {r['size_pt']:>5} pt  '{r['text']}'  at ({r['x_mm']}, {r['y_mm']}) mm  {r['font']}  {r['where'] or ''}")
    extra_bbox = [r for r in ts.get("bbox_below_min", []) if r["gs_size_pt"] is None]
    for r in extra_bbox[:20]:
        p(f"      bbox-only {r['est_size_pt']:>5} pt  '{r['text']}'  at ({r['x_mm']}, {r['y_mm']}) mm  {r['where'] or ''}")
    fc = rep["fonts"]
    p("   fonts    " + "; ".join(f"{f['family']} [{f['type']}]" for f in fc["fonts"]))
    for f in fc["fails"]:
        p(f"   FAIL     {f}")
    for n in fc["non_arial"]:
        p(f"   NOTE     {n}")
    for w in rep.get("layout_warnings", []):
        p(f"   WARN     {w}")
    for w in rep.get("tex", {}).get("missing_glyphs", []):
        p(f"   FAIL     missing glyph {w}")
    for n in rep.get("blind_spots", []):
        p(f"   BLIND    {n}")


def compose(layout_path, workdir=None, name=None, final=None, dpi=150, min_pt=5.0, debug_boxes=False):
    layout_path = Path(layout_path).resolve()
    layout = json.loads(layout_path.read_text())
    name = name or layout.get("name") or layout_path.stem
    workdir = Path(workdir) if workdir else layout_path.parent / f"build_{name}"
    inputs = workdir / "inputs"
    inputs.mkdir(parents=True, exist_ok=True)
    cells = prepare_cells(layout, inputs)
    tex = workdir / f"{name}.tex"
    letters = write_tex(layout, cells, tex)
    texinfo = compile_tex(tex)
    pdf = tex.with_suffix(".pdf")
    png = render_png(pdf, workdir / name, dpi)
    W, H = float(layout["width_mm"]), float(layout["height_mm"])
    pw, ph = page_size_mm(pdf)
    # per-cell native text sizes and fonts
    cell_fonts, blind = {}, []
    for c in cells:
        if c["ext"] == ".pdf":
            fl = pdffonts(inputs / c["local"], c["page"])
            cell_fonts[c["id"]] = fl
            sp = gs_spans(inputs / c["local"], c["page"])
            if sp:
                mn = min(s["size"] for s in sp)
                c["native_min_text_pt"] = round(mn, 2)
                c["placed_min_text_pt"] = round(mn * c["scale"], 2)
            else:
                blind.append(f"cell {c['id']} ({Path(c['file']).name}): no extractable text (outlined or raster "
                             "text cannot be measured; check its type size by eye)")
            c["fonts"] = [font_family(f["name"]) + f" [{f['type']}]" for f in fl]
        else:
            blind.append(f"cell {c['id']} ({Path(c['file']).name}): raster image; text inside it cannot be measured "
                         f"(scale x source font size, or by eye)")
    ts = text_size_check(pdf, min_pt, cells)
    fc = font_check(pdf, cell_fonts)
    rep = dict(name=name, layout=str(layout_path), pdf=str(pdf), png=str(png), requested_mm=[W, H],
               page_size_mm=[round(pw, 3), round(ph, 3)],
               page_size_ok=abs(pw - W) <= TOL_MM and abs(ph - H) <= TOL_MM,
               cells=[{k: v for k, v in c.items()} for c in cells], letters=letters,
               layout_warnings=layout_checks(layout, cells, letters), tex=texinfo, text_size=ts, fonts=fc,
               blind_spots=blind)
    if debug_boxes:
        dtex = workdir / f"{name}_boxes.tex"
        write_tex(layout, cells, dtex, debug_boxes=True)
        compile_tex(dtex)
        rep["png_boxes"] = str(render_png(dtex.with_suffix(".pdf"), workdir / f"{name}_boxes", dpi))
    if final:
        final = Path(final)
        if final.exists():
            rep["final"] = f"NOT WRITTEN: {final} exists (never overwritten)"
        elif not final.parent.is_dir():
            rep["final"] = f"NOT WRITTEN: folder {final.parent} does not exist"
        else:
            shutil.copyfile(pdf, final)
            rep["final"] = str(final)
    (workdir / f"{name}_report.json").write_text(json.dumps(rep, indent=1, default=str))
    summarise(rep)
    if "final" in rep:
        print(f"   final    {rep['final']}")
    return rep


def check_only(pdf, min_pt=5.0, placed_width_mm=None):
    pdf = Path(pdf)
    pw, ph = page_size_mm(pdf)
    scale = (placed_width_mm / pw) if placed_width_mm else 1.0
    rep = dict(pdf=str(pdf), text_size=text_size_check(pdf, min_pt, scale=scale), fonts=font_check(pdf))
    rep["page_size_mm"] = [round(pw * scale, 2), round(ph * scale, 2)]
    rep["requested_mm"], rep["page_size_ok"] = [round(pw, 2), round(ph, 2)], True
    rep["check_scale"] = round(scale, 4)
    summarise(rep)
    if scale != 1.0:
        print(f"   (sizes and positions as printed at {placed_width_mm} mm wide: scale {scale:.4f} of the native "
              f"{pw:.2f} x {ph:.2f} mm)")
    return rep


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("layout", nargs="?")
    ap.add_argument("--workdir"); ap.add_argument("--name"); ap.add_argument("--final")
    ap.add_argument("--dpi", type=int, default=150); ap.add_argument("--min-pt", type=float, default=5.0)
    ap.add_argument("--debug-boxes", action="store_true")
    ap.add_argument("--check", metavar="PDF")
    ap.add_argument("--placed-width-mm", type=float, help="with --check: report sizes as printed at this width")
    ap.add_argument("--panel-info", nargs="+", metavar="FILE")
    ap.add_argument("--json", action="store_true", help="with --check/--panel-info: print JSON")
    a = ap.parse_args()
    if a.panel_info:
        infos = [panel_info(f) for f in a.panel_info]
        if a.json:
            print(json.dumps(infos, indent=1, default=str))
        else:
            for i in infos:
                print(json.dumps(i, default=str))
        return
    if a.check:
        rep = check_only(a.check, a.min_pt, a.placed_width_mm)
        if a.json:
            print(json.dumps(rep, indent=1, default=str))
        return
    if not a.layout:
        ap.error("give a layout JSON, --check PDF or --panel-info FILE")
    compose(a.layout, a.workdir, a.name, a.final, a.dpi, a.min_pt, a.debug_boxes)


if __name__ == "__main__":
    main()
