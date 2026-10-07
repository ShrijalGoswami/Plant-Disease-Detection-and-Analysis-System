"""Validate every $...$ / $$...$$ fragment in outputs/expNN/content.json converts to MathML."""
import json, os, sys
from mathfmt import split_math, latex_to_mathml

LAB = os.path.dirname(os.path.abspath(__file__))
only = [int(a) for a in sys.argv[1:]] or list(range(1, 15))
bad = 0
for n in only:
    p = os.path.join(LAB, "outputs", f"exp{n:02d}", "content.json")
    c = json.load(open(p, encoding="utf-8"))
    n_inline = n_display = 0
    texts = []
    for f in ("theory", "algorithm", "results", "objectives"):
        texts += [(f, t) for t in c.get(f, [])]
    texts.append(("conclusion", c.get("conclusion", "")))
    texts.append(("aim", c.get("aim", "")))
    texts += [("caption", fg.get("caption", "")) for fg in c.get("figures", [])]
    for field, t in texts:
        if t.count("$") % 2:
            print(f"[exp{n:02d}] odd number of $ in {field}: {t[:80]!r}"); bad += 1
        for kind, frag in split_math(t):
            if kind == "text":
                continue
            n_inline += kind == "inline"; n_display += kind == "display"
            try:
                latex_to_mathml(frag)
            except Exception as e:
                print(f"[exp{n:02d}] {field}: cannot convert {frag!r}: {e}"); bad += 1
    print(f"[exp{n:02d}] inline={n_inline} display={n_display}")
print("problems:", bad)
