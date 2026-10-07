"""Shared helpers for LaTeX maths in the lab content: splitting text into
prose / inline / display segments and converting LaTeX to Word OMML."""
import re

import latex2mathml.converter as l2m
from lxml import etree

MML2OMML = r"C:\Program Files\Microsoft Office\root\Office16\MML2OMML.XSL"
_xslt = None

_SPLIT = re.compile(r"(\$\$.+?\$\$|\$[^$]+?\$)", re.S)


def split_math(text):
    """Return [(kind, content)] with kind in {'text', 'inline', 'display'}."""
    out = []
    for part in _SPLIT.split(text):
        if not part:
            continue
        if part.startswith("$$") and part.endswith("$$") and len(part) > 4:
            out.append(("display", part[2:-2].strip()))
        elif part.startswith("$") and part.endswith("$") and len(part) > 2:
            out.append(("inline", part[1:-1].strip()))
        else:
            out.append(("text", part))
    return out


MML_NS = "http://www.w3.org/1998/Math/MathML"
OPEN_CLOSE = {"(": ")", "[": "]", "{": "}", "|": "|", "\u2016": "\u2016", "\u27e8": "\u27e9"}


def preprocess_latex(latex):
    """Small rewrites so the MathML -> OMML path renders nicely in Word."""
    latex = re.sub(r"\\operatorname\{([^}]*)\}", r"\\text{\1}", latex)
    latex = latex.replace(r"\qquad", r"\ \ \ \ ").replace(r"\quad", r"\ \ ")
    # \max, \min, ... come through as identifiers that Word italicises; force upright text
    latex = re.sub(r"\\(max|min|exp|log|ln|sin|cos|tan|det|arg|lim|sup|inf)(?![A-Za-z])", r"\\text{\1}", latex)
    latex = latex.replace(r"\text{arg}\text{max}", r"\text{arg\,max}").replace(r"\text{arg} \text{max}", r"\text{arg\,max}")
    # function names typed without a backslash render italic; make them upright
    latex = re.sub(r"(?<![\\A-Za-z])(max|min|argmax|argmin|exp|log|blur|median|mean|rank|std|atan2|det|rot|trace)(?=\s*[({_\\])",
                   r"\\text{\1}", latex)
    return latex


def latex_to_mathml(latex):
    return l2m.convert(preprocess_latex(latex))


def _fence_to_mfenced(root):
    """Turn <mrow><mo>(</mo> ... <mo>)</mo></mrow> into <mfenced> so Word gets a
    stretchy delimiter (matrices, \\left ... \\right, norms)."""
    tag_mrow, tag_mo = f"{{{MML_NS}}}mrow", f"{{{MML_NS}}}mo"
    tag_mtable = f"{{{MML_NS}}}mtable"
    # matrices written as bare siblings: <mo>[</mo><mtable/><mo>]</mo>
    for mtable in list(root.iter(tag_mtable)):
        prev, nxt = mtable.getprevious(), mtable.getnext()
        if prev is None or nxt is None or prev.tag != tag_mo or nxt.tag != tag_mo:
            continue
        o, c = (prev.text or "").strip(), (nxt.text or "").strip()
        if o not in OPEN_CLOSE or OPEN_CLOSE[o] != c:
            continue
        parent = mtable.getparent()
        fenced = etree.Element(f"{{{MML_NS}}}mfenced", open=o, close=c)
        parent.replace(prev, fenced)
        fenced.append(mtable)
        parent.remove(nxt)
    for mrow in list(root.iter(tag_mrow)):
        kids = list(mrow)
        if len(kids) < 2 or kids[0].tag != tag_mo or kids[-1].tag != tag_mo:
            continue
        o, c = (kids[0].text or "").strip(), (kids[-1].text or "").strip()
        if o not in OPEN_CLOSE or OPEN_CLOSE[o] != c:
            continue
        fenced = etree.Element(f"{{{MML_NS}}}mfenced", open=o, close=c)
        inner = etree.SubElement(fenced, tag_mrow)
        for k in kids[1:-1]:
            inner.append(k)
        mrow.getparent().replace(mrow, fenced)
    # \begin{cases}: prefix brace with no closing partner
    for mrow in list(root.iter(tag_mrow)):
        kids = list(mrow)
        if len(kids) == 2 and kids[0].tag == tag_mo and (kids[0].text or "").strip() == "{" \
                and kids[0].get("fence") == "true" and kids[1].tag == f"{{{MML_NS}}}mtable":
            fenced = etree.Element(f"{{{MML_NS}}}mfenced", open="{", close="")
            fenced.append(kids[1])
            mrow.getparent().replace(mrow, fenced)
    return root


def latex_to_omml(latex, display=False):
    """LaTeX -> MathML (latex2mathml) -> OMML (Office XSLT). Returns an lxml element."""
    global _xslt
    if _xslt is None:
        _xslt = etree.XSLT(etree.parse(MML2OMML))
    mml = etree.fromstring(latex_to_mathml(latex))
    _fence_to_mfenced(mml)
    if display:
        mml.set("display", "block")
    res = _xslt(mml)
    root = res.getroot()
    return root
