"""Find LaTeX fragments whose backslash escapes were mangled (\\t -> tab, \\a -> bell ...)."""
import glob
import json
import re

from mathfmt import split_math

# words that appear when a backslash command lost its backslash
SUSPECT = re.compile(r"(?<![\\A-Za-z])(imes|rac\{|au\b|eta\b|ar\b|ec\{|ext\{|heta\b|lpha\b|ho\b|igma\b|ambda\b|abla\b|orm)")
CTRL = re.compile(r"[\t\r\x00-\x08\x0b\x0c\x0e-\x1f]")
hits = 0
for p in sorted(glob.glob("outputs/exp*/content.json")):
    c = json.load(open(p, encoding="utf-8"))
    texts = [c["aim"], c["conclusion"]] + c["theory"] + c["algorithm"] + c["results"] + c["objectives"] + [f["caption"] for f in c["figures"]]
    for t in texts:
        if CTRL.search(t) or "\n" in t:
            print(p, "CONTROL/NEWLINE:", repr(t[:100])); hits += 1
        for kind, frag in split_math(t):
            if kind != "text" and SUSPECT.search(frag):
                print(p, "SUSPECT:", repr(frag[:120])); hits += 1
print("hits:", hits)
