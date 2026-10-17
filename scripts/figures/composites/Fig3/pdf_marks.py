"""Read-only decoder for matplotlib PDFs: marker placements (XObject 'Do' after a translation 'cm') with their fill
colour, text runs (Tf size + Td/Tm position + string) and filled rectangles ('re'). Used only to check that the
replotted S10 panels draw the same points/labels/bars as the v22 originals. Prints aggregates only."""
import re, sys, zlib


def streams(path):
    b = open(path, 'rb').read()
    out = []
    for m in re.finditer(rb'stream\r?\n', b):
        s = m.end(); e = b.find(b'endstream', s)
        raw = b[s:e]
        try:
            out.append(zlib.decompress(raw).decode('latin-1'))
        except Exception:
            pass
    return out


def content(path):
    """The page content stream: the longest decompressed stream containing 'BT' or ' Do'."""
    ss = [s for s in streams(path) if (' Do' in s or 'BT' in s)]
    return max(ss, key=len)


NUM = r'[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?'


def markers(path):
    """List of (x, y, (r, g, b)) for each 'Do' of a marker XObject, in PDF points (page coordinates). matplotlib writes
    the marker translations relative to each other inside one q..Q block, so they are accumulated."""
    c = content(path)
    out = []; fill = None; stack = []; tx = ty = 0.0
    pat = re.compile(rf'(?P<rg>({NUM})\s+({NUM})\s+({NUM})\s+rg)|(?P<cm>({NUM})\s+({NUM})\s+({NUM})\s+({NUM})\s+({NUM})\s+({NUM})\s+cm)'
                     rf'|(?P<do>/\w+\s+Do)|(?P<q>(?<![\w/])q(?![\w]))|(?P<Q>(?<![\w/])Q(?![\w]))')
    for m in pat.finditer(c):
        if m.group('rg'):
            fill = tuple(float(v) for v in m.group('rg').split()[:3])
        elif m.group('cm'):
            v = [float(z) for z in m.group('cm').split()[:6]]
            tx += v[4]; ty += v[5]                      # pure translations only (markers); rotations are text
        elif m.group('do'):
            out.append((tx, ty, fill))
        elif m.group('q'):
            stack.append((tx, ty, fill))
        elif m.group('Q'):
            tx, ty, fill = stack.pop() if stack else (0.0, 0.0, None)
    return out


def rects(path):
    """Filled axis-aligned rectangles written as 'x0 y0 m x1 y0 l x1 y1 l x0 y1 l h' + B/f: (x, y, w, h, fill)."""
    c = content(path)
    out = []; fill = None
    pat = re.compile(rf'(?P<rg>({NUM})\s+({NUM})\s+({NUM})\s+rg)|(?P<g>(?<![\w.])({NUM})\s+g(?![\w]))'
                     rf'|(?P<box>({NUM})\s+({NUM})\s+m\s+({NUM})\s+({NUM})\s+l\s+'
                     rf'({NUM})\s+({NUM})\s+l\s+({NUM})\s+({NUM})\s+l\s+h\s+[Bf])')
    for m in pat.finditer(c):
        if m.group('rg'):
            fill = tuple(float(v) for v in m.group('rg').split()[:3])
        elif m.group('g'):                              # grey fill operator (added for Fig 3: matplotlib writes greys as 'v g')
            v = float(m.group('g').split()[0]); fill = (v, v, v)
        else:
            v = [float(z) for z in re.findall(NUM, m.group('box'))]
            x0, y0, x1, y1b, x2, y2, x3, y3 = v
            if abs(y0 - y1b) < 1e-6 and abs(x1 - x2) < 1e-6 and abs(y2 - y3) < 1e-6 and abs(x3 - x0) < 1e-6:
                out.append((x0, y0, x1 - x0, y2 - y0, fill))
    return out


if __name__ == '__main__':
    p = sys.argv[1]
    mk = markers(p)
    print(len(mk), 'markers;', len(rects(p)), 'rects')
