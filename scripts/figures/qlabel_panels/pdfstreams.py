"""Minimal PDF stream extractor (zlib) for matplotlib-generated PDFs."""
import re, zlib, sys

def objects(path):
    data = open(path, 'rb').read()
    out = {}
    for m in re.finditer(rb'(\d+) 0 obj\s*(.*?)endobj', data, re.S):
        num = int(m.group(1)); body = m.group(2)
        sm = re.search(rb'stream\r?\n(.*?)\r?\nendstream', body, re.S)
        d = body[:sm.start()] if sm else body
        s = None
        if sm:
            raw = sm.group(1)
            if b'/FlateDecode' in d:
                try:
                    s = zlib.decompress(raw)
                except Exception:
                    s = zlib.decompressobj().decompress(raw)
            else:
                s = raw
        out[num] = (d, s)
    return out

if __name__ == '__main__':
    objs = objects(sys.argv[1])
    for n, (d, s) in sorted(objs.items()):
        print(n, d[:300].decode('latin1').replace('\n', ' '), '| stream len', None if s is None else len(s))
