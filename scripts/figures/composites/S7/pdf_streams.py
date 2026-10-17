"""Tiny helper: decompress the content streams of a matplotlib PDF (read-only) for inspection."""
import re, sys, zlib
def streams(path):
    b = open(path, 'rb').read()
    out = []
    for m in re.finditer(rb'stream\r?\n', b):
        s = m.end(); e = b.find(b'endstream', s)
        raw = b[s:e]
        try:
            out.append(zlib.decompress(raw))
        except Exception:
            out.append(raw)
    return out
if __name__ == '__main__':
    for i, s in enumerate(streams(sys.argv[1])):
        print('#', i, len(s), s[:int(sys.argv[2]) if len(sys.argv) > 2 else 300])
