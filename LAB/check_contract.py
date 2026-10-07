"""Validate outputs/expNN/content.json + files against CONTRACT.md."""
import json
import os
import re
import sys

LAB = os.path.dirname(os.path.abspath(__file__))
REQUIRED = ["number", "title", "aim", "objectives", "theory", "tools", "dataset", "algorithm",
            "code_file", "figures", "results", "conclusion"]
EXPECTED_TITLES = {
    1: "Implement image preprocessing and Edge detection",
    2: "Implement camera calibration methods",
    3: "Implement Projection",
    4: "Determine depth map from Stereo pair",
    5: "Construct 3D model from Stereo pair",
    6: "Implement Segmentation methods",
    7: "Construct 3D model from defocus image",
    8: "Construct 3D model from Images",
    9: "Implement optical flow method",
    10: "Implement object detection and tracking from video",
    11: "Face detection and Recognition",
    12: "Object detection from dynamic Background for Surveillance",
    13: "Content based video retrieval",
    14: "Construct 3D model from single image",
}

only = [int(a) for a in sys.argv[1:]] or list(range(1, 15))
problems = 0
for n in only:
    d = os.path.join(LAB, "outputs", f"exp{n:02d}")
    cpath = os.path.join(d, "content.json")
    tag = f"exp{n:02d}"
    if not os.path.exists(cpath):
        print(f"[{tag}] MISSING content.json")
        problems += 1
        continue
    try:
        c = json.load(open(cpath, encoding="utf-8"))
    except Exception as e:
        print(f"[{tag}] content.json invalid JSON: {e}")
        problems += 1
        continue
    issues = []
    for k in REQUIRED:
        if k not in c:
            issues.append(f"missing key {k}")
    if c.get("number") != n:
        issues.append(f"number {c.get('number')} != {n}")
    if c.get("title") != EXPECTED_TITLES[n]:
        issues.append(f"title mismatch: {c.get('title')!r}")
    code = os.path.join(LAB, c.get("code_file", ""))
    if not os.path.isfile(code):
        issues.append(f"code file missing {c.get('code_file')}")
    else:
        nlines = sum(1 for _ in open(code, encoding="utf-8", errors="replace"))
        if nlines > 230:
            issues.append(f"code too long: {nlines} lines")
        src = open(code, encoding="utf-8", errors="replace").read()
        if "imshow(" in src and "cv2.imshow" in src:
            issues.append("cv2.imshow used")
    for f in c.get("figures", []):
        fp = os.path.join(d, f.get("file", ""))
        if not os.path.isfile(fp):
            issues.append(f"figure missing {f.get('file')}")
        elif os.path.getsize(fp) < 5000:
            issues.append(f"figure suspiciously small {f.get('file')}")
        if not f.get("caption", "").startswith(f"Figure {n}."):
            issues.append(f"caption format: {f.get('caption','')[:40]!r}")
    if not (2 <= len(c.get("figures", [])) <= 6):
        issues.append(f"{len(c.get('figures', []))} figures")
    if not os.path.isfile(os.path.join(d, "console.txt")):
        issues.append("console.txt missing")
    else:
        txt = open(os.path.join(d, "console.txt"), encoding="utf-8", errors="replace").read()
        if "Traceback" in txt or "Error" in txt.split("\n")[-3:]:
            issues.append("console.txt contains a Traceback/Error")
    words = sum(len(p.split()) for p in c.get("theory", []) if not p.startswith("## "))
    if not (80 <= words <= 200):
        issues.append(f"theory word count {words}")
    if not any(re.search(r"\d", r) for r in c.get("results", [])):
        issues.append("results contain no numbers")
    if "viva" in c:
        issues.append("viva key still present")
    if not (5 <= len(c.get("algorithm", [])) <= 12):
        issues.append(f"algorithm steps {len(c.get('algorithm', []))}")
    if not c.get("aim", "").startswith("To"):
        issues.append("aim does not start with 'To'")
    blob = json.dumps(c, ensure_ascii=False)
    if "—" in blob:
        issues.append("em dash present")
    status = "OK " if not issues else "BAD"
    print(f"[{tag}] {status} figs={len(c.get('figures', []))} theory_words={words} :: " + ("; ".join(issues) if issues else "all checks passed"))
    problems += len(issues)
print("total problems:", problems)
