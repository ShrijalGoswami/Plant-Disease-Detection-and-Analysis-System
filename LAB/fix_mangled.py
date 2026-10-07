"""Repair backslash escapes that were mangled into control characters."""
import glob
import json

REPL = {"\t": "\\t", "\x07": "\\a", "\x08": "\\b", "\x0c": "\\f", "\x0b": "\\v", "\r": "\\r"}


def fix(x):
    if isinstance(x, str):
        for k, v in REPL.items():
            x = x.replace(k, v)
        return x
    if isinstance(x, list):
        return [fix(i) for i in x]
    if isinstance(x, dict):
        return {k: fix(v) for k, v in x.items()}
    return x


for p in sorted(glob.glob("outputs/exp*/content.json")):
    c = json.load(open(p, encoding="utf-8"))
    c2 = fix(c)
    if c2 != c:
        json.dump(c2, open(p, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
        print("fixed", p)
