"""Read-only: extract marker centres (with fill colour) and text spans from a matplotlib PDF content stream."""
import re, zlib, numpy as np

def content(path):
    b = open(path, 'rb').read(); out = []
    for m in re.finditer(rb'stream\r?\n', b):
        s = m.end(); e = b.find(b'endstream', s)
        try:
            d = zlib.decompress(b[s:e])
        except Exception:
            continue
        if b' Do' in d or b'BT' in d or b' re' in d:
            out.append(d.decode('latin1'))
    return out

def _tokens(s):
    # drop text objects (BT..ET) and inline strings
    s = re.sub(r'BT.*?ET', ' ', s, flags=re.S)
    return s.split()

def markers(path, max_d=30.0):
    """(x, y, rgb) for every XObject marker stamp (cumulative cm translations inside q..Q) and every inline closed
    circle path (4 Bezier arcs, width == height), in PDF points."""
    pts = []
    for s in content(path):
        ops = []; col = None; ctm = np.eye(3); stack = []; path = []
        for t in _tokens(s):
            try:
                ops.append(float(t)); continue
            except ValueError:
                pass
            if t == 'rg':
                col = tuple(round(v, 3) for v in ops[-3:])
            elif t == 'q':
                stack.append(ctm.copy())
            elif t == 'Q':
                ctm = stack.pop() if stack else np.eye(3)
            elif t == 'cm':
                a, b, c, d, e, f = ops[-6:]
                M = np.array([[a, b, 0], [c, d, 0], [e, f, 1]]); ctm = M @ ctm
            elif t == 'Do':
                pts.append((ctm[2, 0], ctm[2, 1], col))
            elif t == 'm':
                path = [tuple(ops[-2:])]
            elif t == 'c':
                v = ops[-6:]; path += [(v[0], v[1]), (v[2], v[3]), (v[4], v[5])]
            elif t == 'l':
                path.append(tuple(ops[-2:]))
            elif t == 'h':
                if len(path) in (13, 25, 33):
                    P = np.array(path); w = np.ptp(P[:, 0]); h = np.ptp(P[:, 1])
                    if w < max_d and abs(w - h) < 0.05 * max(w, h, 1e-9):
                        q = np.array([(P[:, 0].max() + P[:, 0].min()) / 2, (P[:, 1].max() + P[:, 1].min()) / 2, 1]) @ ctm
                        pts.append((q[0], q[1], col))
                path = []
            elif t.startswith('/'):
                ops.append(t); continue
            ops = [] if t not in ('rg',) else []
    return pts

def texts(path):
    out = []
    for s in content(path):
        for m in re.finditer(r'(-?[\d.]+) (-?[\d.]+) (-?[\d.]+) (-?[\d.]+) (-?[\d.]+) (-?[\d.]+) cm\s*BT\s*/F\d+ (-?[\d.]+) Tf\s*0 0 Td\s*\[(.*?)\] TJ', s, re.S):
            txt = ''.join(re.findall(r'\((.*?)\)', m.group(8))).replace(' ', '').replace('"\x12', '\u2212').replace('\x00', '')
            out.append(dict(text=txt, x=float(m.group(5)), y=float(m.group(6)), size=float(m.group(7)),
                            rot=float(m.group(2)) != 0))
    return out
