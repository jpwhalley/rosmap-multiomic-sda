"""Minimal read-only parser for single-page matplotlib PDFs (content stream: text blocks, filled rects)."""
import re, zlib

def page_stream(path):
    data = open(path, 'rb').read()
    mb = re.search(rb'/MediaBox \[ ([\d.\s]+) \]', data)
    media = [float(v) for v in mb.group(1).split()]
    cm = re.search(rb'/Contents (\d+) 0 R', data)
    num = cm.group(1)
    m = re.search(rb'\n' + num + rb' 0 obj\n(.*?)stream\n', data, re.S)
    start = m.end()
    end = data.find(b'endstream', start)
    return media, zlib.decompress(data[start:end]).decode('latin1')

def _decode_tj(arr):
    # strings inside (...) with PDF escapes; Identity-H 2-byte UTF-16BE (matplotlib type42)
    out = []
    i = 0
    while i < len(arr):
        if arr[i] == '(':
            depth, j, buf = 1, i + 1, []
            while depth:
                ch = arr[j]
                if ch == '\\':
                    nxt = arr[j + 1]
                    if nxt in '()\\':
                        buf.append(nxt); j += 2; continue
                    if nxt.isdigit():
                        oc = re.match(r'[0-7]{1,3}', arr[j + 1:]).group(0)
                        buf.append(chr(int(oc, 8))); j += 1 + len(oc); continue
                    buf.append({'n': '\n', 'r': '\r', 't': '\t', 'b': '\b', 'f': '\f'}.get(nxt, nxt)); j += 2; continue
                if ch == '(':
                    depth += 1
                elif ch == ')':
                    depth -= 1
                    if depth == 0:
                        break
                buf.append(ch); j += 1
            out.append(''.join(buf)); i = j + 1
        else:
            i += 1
    raw = ''.join(out).encode('latin1')
    if len(raw) % 2 == 0 and raw[::2].count(0) == len(raw) // 2:
        return raw.decode('utf-16-be')
    return raw.decode('latin1')

def texts(stream):
    res = []
    for m in re.finditer(r'1 0 -?0 1 ([\d.\-]+) ([\d.\-]+) cm\s*BT\s*/F\d+ ([\d.]+) Tf\s*0 0 Td\s*(.*?)\s*ET', stream, re.S):
        x, y, size, body = float(m.group(1)), float(m.group(2)), float(m.group(3)), m.group(4)
        tj = re.search(r'\[(.*)\]\s*TJ', body, re.S)
        s = _decode_tj(tj.group(1)) if tj else ''
        res.append(dict(x=x, y=y, size=size, text=s))
    return res

def filled_rects(stream):
    """Rects drawn as 4-point paths followed by B or f; fill colour = last rg/g operator before the path."""
    res = []
    pat = re.compile(r'([\d.]+) ([\d.]+) m\n([\d.]+) ([\d.]+) l\n([\d.]+) ([\d.]+) l\n([\d.]+) ([\d.]+) l\nh\n\n(B|f)\b')
    colpat = re.compile(r'(?:([\d.]+) ([\d.]+) ([\d.]+) rg|(?<![\d.])([\d.]+) g(?![a-zA-Z]))')
    for m in pat.finditer(stream):
        cols = list(colpat.finditer(stream, 0, m.start()))
        c = cols[-1]
        col = (float(c.group(1)), float(c.group(2)), float(c.group(3))) if c.group(1) else (float(c.group(4)),) * 3
        xs = [float(m.group(k)) for k in (1, 3, 5, 7)]
        ys = [float(m.group(k)) for k in (2, 4, 6, 8)]
        res.append(dict(color=col, x0=min(xs), x1=max(xs), y0=min(ys), y1=max(ys), op=m.group(9)))
    return res
