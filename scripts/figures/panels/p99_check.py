"""QA over every panel PDF in a folder tree: fonts (embedded, TrueType/Type 42 Arial, no Type 3), clipping (ink bounding box
touching the page edge), withdrawn or forbidden terms in the text layer, and donor IDs in the text layer (checked against
the SDA donor list; IDs are never printed). Usage: p99_check.py <root> [<root> ...]"""
import sys, os, re, subprocess, glob
sys.path.insert(0, os.path.dirname(__file__))
from vp_common import donors

FORBID = ["CACNA1I", "ENTHD1", "PRDM4", "pluripoten", "Protein", "protein", "mTOR", "AXL", "FLT4", "trans", "♂", "♀", "precursor",
          "Importantly", "Notably", "Interestingly", "—"]
D = set(donors())
rows = []
for root in sys.argv[1:]:
    for f in sorted(glob.glob(f"{root}/**/*.pdf", recursive=True)):
        fonts = subprocess.run(["pdffonts", f], capture_output=True, text=True).stdout.splitlines()[2:]
        names = sorted({l.split()[0] for l in fonts})
        types = sorted({" ".join(l.split()[1:3]) for l in fonts})
        emb_no = [l for l in fonts if len(l.split()) > 4 and l.split()[-5] == "no"]
        t3 = any("Type 3" in l for l in fonts)
        nonarial = [n for n in names if "Arial" not in n]
        txt = subprocess.run(["pdftotext", f, "-"], capture_output=True, text=True).stdout
        bad = [w for w in FORBID if w in txt]
        ids = sum(1 for tok in set(re.findall(r"\d{5,9}", txt)) if tok in D)
        bb = subprocess.run(["gs", "-q", "-dNOPAUSE", "-dBATCH", "-sDEVICE=bbox", f], capture_output=True, text=True).stderr
        hb = [float(x) for x in re.search(r"%%HiResBoundingBox: ([\d. ]+)", bb).group(1).split()]
        info = subprocess.run(["pdfinfo", f], capture_output=True, text=True).stdout
        pw, ph = [float(x) for x in re.search(r"Page size:\s+([\d.]+) x ([\d.]+)", info).group(1, 2)]
        edge = min(hb[0], hb[1], pw - hb[2], ph - hb[3])
        rows.append((os.path.relpath(f, root), names, types, t3, len(emb_no), nonarial, bad, ids, round(edge, 1), round(pw / 72, 2), round(ph / 72, 2)))
ok = True
for r in rows:
    flag = []
    if r[3]: flag.append("TYPE3")
    if r[4]: flag.append("NOT-EMBEDDED")
    if r[5]: flag.append("NON-ARIAL:" + ",".join(r[5]))
    if r[6]: flag.append("TERMS:" + ",".join(r[6]))
    if r[7]: flag.append(f"DONOR-ID-TOKENS:{r[7]}")
    if r[8] < 2: flag.append(f"EDGE {r[8]} pt")
    ok &= not flag
    print(f"{r[0]}  [{r[9]} x {r[10]} in]  fonts={','.join(r[1]) or '-'}  {' '.join(flag) or 'OK'}")
print("ALL OK" if ok else "FLAGS PRESENT")
