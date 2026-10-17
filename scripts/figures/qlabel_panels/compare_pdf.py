"""Compare two matplotlib PDFs at the content-stream level.

Reports: MediaBox, fonts (BaseFont + /W widths), marker XObject definitions, decoded text
strings with their positions, every non-marker drawing token, and the stripplot marker
placements (dy exactly, dx exactly) per collection.

Also usable as a module: parse(path) -> dict, markers_abs(parsed) -> absolute PDF-pt positions.
"""
import re
import sys
from decimal import Decimal

HERE = __import__('os').path.dirname(__import__('os').path.abspath(__file__))
sys.path.insert(0, HERE)
from pdfstreams import objects  # noqa: E402


def lex(s):
    """Tokenise a PDF content stream (bytes)."""
    toks, i, n = [], 0, len(s)
    ws = b' \t\r\n\x0c\x00'
    while i < n:
        c = s[i:i + 1]
        if c in (b' ', b'\t', b'\r', b'\n', b'\x0c', b'\x00'):
            i += 1
            continue
        if c == b'(':
            depth, j, buf = 1, i + 1, bytearray()
            while depth:
                ch = s[j:j + 1]
                if ch == b'\\':
                    nxt = s[j + 1:j + 2]
                    esc = {b'n': b'\n', b'r': b'\r', b't': b'\t', b'b': b'\b', b'f': b'\f'}
                    if nxt in esc:
                        buf += esc[nxt]; j += 2
                    elif nxt.isdigit():
                        m = re.match(rb'[0-7]{1,3}', s[j + 1:j + 4])
                        buf.append(int(m.group(0), 8)); j += 1 + len(m.group(0))
                    else:
                        buf += nxt; j += 2
                    continue
                if ch == b'(':
                    depth += 1
                elif ch == b')':
                    depth -= 1
                    if depth == 0:
                        break
                buf += ch
                j += 1
            toks.append(('str', bytes(buf)))
            i = j + 1
            continue
        if c in (b'[', b']'):
            toks.append(('op', c.decode()))
            i += 1
            continue
        j = i
        while j < n and s[j:j + 1] not in ws and s[j:j + 1] not in (b'(', b'[', b']'):
            j += 1
        toks.append(('tok', s[i:j].decode('latin1')))
        i = j
    return toks


def parse(path):
    objs = objects(path)
    page = next(d for d, s in objs.values() if b'/Type /Page ' in d)
    cnum = int(re.search(rb'/Contents (\d+) 0 R', page).group(1))
    mediabox = re.search(rb'/MediaBox \[([^\]]*)\]', page).group(1).decode().split()
    content = objs[cnum][1]
    fonts = [re.search(rb'/BaseFont /(\S+)', d).group(1).decode() for d, s in objs.values()
             if b'/Type /Font' in d and b'/BaseFont' in d]
    xobj = {}
    xmap = next((d for d, s in objs.values() if re.match(rb'<< /M\d+ \d+ 0 R', d.strip())), b'')
    for name, num in re.findall(rb'/(M\d+) (\d+) 0 R', xmap):
        d, st = objs[int(num)]
        xobj[name.decode()] = (re.sub(rb'/Length \d+ ', b'', d).decode(), st)
    toks = lex(content)
    # markers: 1 0 0 1 dx dy cm /Mx Do
    markers, other = {}, []
    k = 0
    while k < len(toks):
        t = [x[1] for x in toks[k:k + 8]]
        if (len(t) == 8 and t[0:4] == ['1', '0', '0', '1'] and t[6] == 'cm'
                and isinstance(t[7], str) and t[7].startswith('/M')
                and k + 8 < len(toks) and toks[k + 8][1] == 'Do'):
            markers.setdefault(t[7][1:], []).append((t[4], t[5]))
            other.append(('MARK', t[7][1:]))
            k += 9
            continue
        other.append(toks[k])
        k += 1
    # collapse runs of MARK
    collapsed = []
    for t in other:
        if t[0] == 'MARK' and collapsed and collapsed[-1][0] == 'MARKS' and collapsed[-1][1] == t[1]:
            collapsed[-1] = ('MARKS', t[1], collapsed[-1][2] + 1)
        elif t[0] == 'MARK':
            collapsed.append(('MARKS', t[1], 1))
        else:
            collapsed.append(t)
    texts = []
    for k, t in enumerate(toks):
        if t == ('tok', 'TJ'):
            j = k - 1
            while toks[j] != ('op', '['):
                j -= 1
            chunk = b''.join(x[1] for x in toks[j:k] if x[0] == 'str')
            txt = chunk.decode('utf-16-be', errors='replace')
            # position: the 'cm' before BT
            m = k
            while toks[m] != ('tok', 'cm'):
                m -= 1
            pos = [x[1] for x in toks[m - 6:m]]
            texts.append((txt, pos[4], pos[5], pos[0:4]))
    W = [d.decode() for d, s in objs.values() if d.strip().startswith(b'[ 32')]
    return dict(mediabox=mediabox, fonts=sorted(set(fonts)), xobj=xobj, markers=markers,
                other=collapsed, texts=texts, W=W, content=content)


def markers_abs(p):
    """Absolute PDF-point positions per marker collection (exact decimal cumulative sums)."""
    out = {}
    for name, lst in p['markers'].items():
        x = y = Decimal(0)
        pts = []
        for dx, dy in lst:
            x += Decimal(dx)
            y += Decimal(dy)
            pts.append((x, y))
        out[name] = pts
    return out


def compare(a_path, b_path, verbose=True, ignore_text=None):
    a, b = parse(a_path), parse(b_path)
    rep = {}
    rep['mediabox'] = a['mediabox'] == b['mediabox']
    rep['fonts'] = (a['fonts'], b['fonts'])
    rep['xobj_identical'] = ({k: v[1] for k, v in a['xobj'].items()} ==
                             {k: v[1] for k, v in b['xobj'].items()})
    rep['texts_a'] = [t[0] for t in a['texts']]
    rep['texts_b'] = [t[0] for t in b['texts']]
    tdiff = []
    for ta, tb in zip(a['texts'], b['texts']):
        if ta != tb:
            tdiff.append((ta, tb))
    if len(a['texts']) != len(b['texts']):
        tdiff.append(('N texts', len(a['texts']), len(b['texts'])))
    rep['text_diffs'] = tdiff
    # non-marker tokens
    oa, ob = a['other'], b['other']
    nd = []
    import difflib
    sm = difflib.SequenceMatcher(a=[repr(x) for x in oa], b=[repr(x) for x in ob], autojunk=False)
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag != 'equal':
            nd.append((tag, oa[i1:i2], ob[j1:j2]))
    rep['nonmarker_token_diffs'] = nd
    # markers
    md = {}
    for name in sorted(set(a['markers']) | set(b['markers'])):
        la, lb = a['markers'].get(name, []), b['markers'].get(name, [])
        md[name] = dict(n=(len(la), len(lb)),
                        dy_identical=[d[1] for d in la] == [d[1] for d in lb],
                        dx_identical=[d[0] for d in la] == [d[0] for d in lb],
                        n_dx_diff=sum(1 for p, q in zip(la, lb) if p[0] != q[0]),
                        n_dy_diff=sum(1 for p, q in zip(la, lb) if p[1] != q[1]))
        aa, bb = markers_abs(a).get(name, []), markers_abs(b).get(name, [])
        if aa and bb and len(aa) == len(bb):
            md[name]['max_abs_dx_pt'] = float(max(abs(p[0] - q[0]) for p, q in zip(aa, bb)))
            md[name]['max_abs_dy_pt'] = float(max(abs(p[1] - q[1]) for p, q in zip(aa, bb)))
    rep['markers'] = md
    rep['W'] = (a['W'], b['W'])
    return rep


if __name__ == '__main__':
    import pprint
    pprint.pprint(compare(sys.argv[1], sys.argv[2]), width=150)
