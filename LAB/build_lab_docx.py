"""
Assemble the final Computer Vision lab file.

Reads the original "index page.docx" (cover + index table) and appends one
section per experiment built from outputs/expNN/content.json, the script in
code/, the console output and the figure PNGs.

Usage:  python build_lab_docx.py [content_root] [output.docx]
"""
import json
import os
import sys

import docx
from docx.enum.section import WD_SECTION
from docx.enum.style import WD_STYLE_TYPE
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_BREAK, WD_LINE_SPACING
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor
from PIL import Image

from mathfmt import split_math, latex_to_omml

LAB = os.path.dirname(os.path.abspath(__file__))
CONTENT_ROOT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(LAB, "outputs")
OUT_PATH = sys.argv[2] if len(sys.argv) > 2 else os.path.join(LAB, "CSE3010_Computer_Vision_Lab_File.docx")
TEMPLATE = os.path.join(LAB, "index page.docx")

BODY_FONT = "Calibri"
CODE_FONT = "Consolas"
ACCENT = RGBColor(0x1F, 0x3A, 0x5F)
GREY = RGBColor(0x59, 0x59, 0x59)
MAX_FIG_W = 6.2      # inches (A4 with 1 in margins gives 6.27 in)
MAX_FIG_H = 7.6      # inches
MAX_CONSOLE_LINES = 18
MAX_FIG_PX = 1200    # maximum embedded figure width in pixels


# --------------------------------------------------------------------------- helpers
def set_cell_shading(cell, hex_fill):
    tc_pr = cell._tc.get_or_add_tcPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:val"), "clear")
    shd.set(qn("w:color"), "auto")
    shd.set(qn("w:fill"), hex_fill)
    tc_pr.append(shd)


def set_paragraph_shading(paragraph, hex_fill):
    p_pr = paragraph._p.get_or_add_pPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:val"), "clear")
    shd.set(qn("w:color"), "auto")
    shd.set(qn("w:fill"), hex_fill)
    p_pr.append(shd)


def set_paragraph_border(paragraph, color="BFBFBF", size=4):
    p_pr = paragraph._p.get_or_add_pPr()
    borders = OxmlElement("w:pBdr")
    for side in ("top", "left", "bottom", "right"):
        el = OxmlElement(f"w:{side}")
        el.set(qn("w:val"), "single")
        el.set(qn("w:sz"), str(size))
        el.set(qn("w:space"), "4")
        el.set(qn("w:color"), color)
        borders.append(el)
    p_pr.append(borders)


def set_outline_level(style, level):
    p_pr = style.element.get_or_add_pPr()
    el = OxmlElement("w:outlineLvl")
    el.set(qn("w:val"), str(level))
    p_pr.append(el)


def add_field(run, instr):
    """Insert a Word field (e.g. PAGE) into a run."""
    fld_begin = OxmlElement("w:fldChar")
    fld_begin.set(qn("w:fldCharType"), "begin")
    instr_el = OxmlElement("w:instrText")
    instr_el.set(qn("xml:space"), "preserve")
    instr_el.text = instr
    fld_sep = OxmlElement("w:fldChar")
    fld_sep.set(qn("w:fldCharType"), "separate")
    txt = OxmlElement("w:t")
    txt.text = "1"
    fld_end = OxmlElement("w:fldChar")
    fld_end.set(qn("w:fldCharType"), "end")
    for el in (fld_begin, instr_el, fld_sep, txt, fld_end):
        run._r.append(el)


def make_styles(d):
    styles = d.styles

    def para_style(name, size, bold=False, color=None, before=0, after=6, keep_next=False,
                   italic=False, font=BODY_FONT, align=None, page_break=False, outline=None,
                   line=1.1):
        st = styles.add_style(name, WD_STYLE_TYPE.PARAGRAPH)
        st.base_style = styles["Normal"]
        st.font.name = font
        st.element.rPr.rFonts.set(qn("w:eastAsia"), font)
        st.font.size = Pt(size)
        st.font.bold = bold
        st.font.italic = italic
        if color is not None:
            st.font.color.rgb = color
        pf = st.paragraph_format
        pf.space_before = Pt(before)
        pf.space_after = Pt(after)
        pf.keep_with_next = keep_next
        pf.page_break_before = page_break
        pf.line_spacing_rule = WD_LINE_SPACING.MULTIPLE
        pf.line_spacing = line
        if align is not None:
            pf.alignment = align
        if outline is not None:
            set_outline_level(st, outline)
        return st

    para_style("LabExpNo", 20, bold=True, color=ACCENT, before=0, after=2, keep_next=True,
               align=WD_ALIGN_PARAGRAPH.CENTER, page_break=True, outline=0)
    para_style("LabExpTitle", 15, bold=True, color=RGBColor(0, 0, 0), before=0, after=10,
               keep_next=True, align=WD_ALIGN_PARAGRAPH.CENTER)
    para_style("LabH2", 12.5, bold=True, color=ACCENT, before=10, after=4, keep_next=True, outline=1)
    para_style("LabH3", 11, bold=True, color=RGBColor(0x26, 0x26, 0x26), before=6, after=3, keep_next=True)
    para_style("LabBody", 11, after=6, align=WD_ALIGN_PARAGRAPH.JUSTIFY, line=1.15)
    para_style("LabBullet", 11, after=3, align=WD_ALIGN_PARAGRAPH.LEFT, line=1.12)
    para_style("LabCode", 8.5, after=0, font=CODE_FONT, line=1.0)
    para_style("LabConsole", 8, after=0, font=CODE_FONT, line=1.0)
    para_style("LabCaption", 10, italic=True, color=GREY, before=2, after=10,
               align=WD_ALIGN_PARAGRAPH.CENTER)
    para_style("LabFigure", 11, before=6, after=0, keep_next=True, align=WD_ALIGN_PARAGRAPH.CENTER)
    para_style("LabSmall", 9.5, color=GREY, after=4)
    para_style("LabEquation", 11.5, before=4, after=8, align=WD_ALIGN_PARAGRAPH.CENTER)
    # indent for bullets
    styles["LabBullet"].paragraph_format.left_indent = Inches(0.3)
    styles["LabBullet"].paragraph_format.first_line_indent = Inches(-0.2)
    styles["LabCode"].paragraph_format.left_indent = Inches(0.05)
    styles["LabConsole"].paragraph_format.left_indent = Inches(0.05)


def body(d, text, style="LabBody"):
    p = d.add_paragraph(style=style)
    add_runs_with_bold(p, text)
    return p


def add_runs_with_bold(p, text):
    """Support **bold** spans and $inline$ LaTeX maths inside a text string."""
    parts = text.split("**")
    for i, part in enumerate(parts):
        if not part:
            continue
        for kind, frag in split_math(part):
            if kind == "text":
                r = p.add_run(frag)
                if i % 2 == 1:
                    r.bold = True
            else:
                add_inline_math(p, frag)


def add_inline_math(p, latex):
    try:
        omath = latex_to_omml(latex)
    except Exception as e:  # fall back to plain text rather than lose content
        print("WARN inline maths failed:", latex, e)
        p.add_run(safe_text(latex))
        return
    p._p.append(omath)


def safe_text(s):
    return "".join(ch for ch in s if ch >= " " or ch in "\t\n")


def display_equation(d, latex):
    """A centred display equation in its own paragraph (m:oMathPara)."""
    p = d.add_paragraph(style="LabEquation")
    try:
        omath = latex_to_omml(latex, display=True)
    except Exception as e:
        print("WARN display maths failed:", latex, e)
        p.add_run(safe_text(latex))
        return p
    if omath.tag == qn("m:oMathPara"):
        para = omath
    else:
        para = OxmlElement("m:oMathPara")
        para.append(omath)
    ppr = para.find(qn("m:oMathParaPr"))
    if ppr is None:
        ppr = OxmlElement("m:oMathParaPr")
        para.insert(0, ppr)
    jc = OxmlElement("m:jc")
    jc.set(qn("m:val"), "center")
    ppr.append(jc)
    p._p.append(para)
    return p


def body_or_equation(d, text, style="LabBody"):
    """A paragraph, or a display equation when the whole item is $$...$$."""
    segs = split_math(text)
    if len(segs) == 1 and segs[0][0] == "display":
        return display_equation(d, segs[0][1])
    return body(d, text, style)


def bullet(d, text, marker="•"):
    p = d.add_paragraph(style="LabBullet")
    p.add_run(marker + "\t")
    add_runs_with_bold(p, text)
    return p


def numbered(d, idx, text):
    return bullet(d, text, marker=f"{idx}.")


def heading2(d, text):
    return d.add_paragraph(text, style="LabH2")


def heading3(d, text):
    return d.add_paragraph(text, style="LabH3")


def code_block(d, code_text, style="LabCode", fill="F3F3F3", max_lines=None):
    lines = code_text.rstrip("\n").split("\n")
    truncated = False
    if max_lines is not None and len(lines) > max_lines:
        lines = lines[:max_lines]
        truncated = True
    n = len(lines)
    for i, line in enumerate(lines):
        p = d.add_paragraph(style=style)
        p.add_run(line.replace("\t", "    ") if line else " ")
        set_paragraph_shading(p, fill)
        # keep the block together in small groups so it does not split awkwardly
        if i < n - 1:
            p.paragraph_format.keep_with_next = (i % 6 != 5)
    if truncated:
        p = d.add_paragraph(style=style)
        p.add_run("... (output truncated) ...")
        set_paragraph_shading(p, fill)


def figure(d, path, caption):
    path = trimmed_figure(path)
    with Image.open(path) as im:
        w_px, h_px = im.size
    aspect = h_px / w_px
    width = MAX_FIG_W
    if width * aspect > MAX_FIG_H:
        width = MAX_FIG_H / aspect
    # do not blow up small images
    natural_w = w_px / 150.0
    width = min(width, max(natural_w, 3.5))
    p = d.add_paragraph(style="LabFigure")
    p.add_run().add_picture(path, width=Inches(width))
    cap = d.add_paragraph(style="LabCaption")
    cap.add_run(caption)


def info_table(d, rows):
    t = d.add_table(rows=0, cols=2)
    t.style = d.styles["Table Grid"]
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    for k, v in rows:
        cells = t.add_row().cells
        cells[0].width = Inches(1.7)
        cells[1].width = Inches(4.5)
        set_cell_shading(cells[0], "E8EEF5")
        p0 = cells[0].paragraphs[0]
        r0 = p0.add_run(k)
        r0.bold = True
        r0.font.size = Pt(10.5)
        r0.font.name = BODY_FONT
        p1 = cells[1].paragraphs[0]
        r1 = p1.add_run(v)
        r1.font.size = Pt(10.5)
        r1.font.name = BODY_FONT
        for c in cells:
            c.paragraphs[0].paragraph_format.space_after = Pt(2)
            c.paragraphs[0].paragraph_format.space_before = Pt(2)
    d.add_paragraph(style="LabSmall")
    return t


def read_text(path, default=""):
    if not os.path.exists(path):
        return default
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        return f.read()


NOISE_PATTERNS = ("FutureWarning", "warnings.warn", "Using cache found", "Loading weights:",
                  "UserWarning", "DeprecationWarning", "site-packages")


def clean_console(text):
    """Drop library warning lines and progress noise from captured stdout."""
    lines = [ln.rstrip() for ln in text.split("\n")]
    lines = [ln for ln in lines if not any(p in ln for p in NOISE_PATTERNS)]
    # collapse runs of blank lines
    out = []
    for ln in lines:
        if ln.strip() == "" and out and out[-1].strip() == "":
            continue
        out.append(ln)
    return "\n".join(out).strip()


TRIM_DIR = os.path.join(os.environ.get("TEMP", LAB), "lab_docx_trimmed")


def trimmed_figure(path, pad=12):
    """Return the path of a copy of the figure with uniform white margins removed."""
    from PIL import ImageChops
    os.makedirs(TRIM_DIR, exist_ok=True)
    out = os.path.join(TRIM_DIR, os.path.basename(os.path.dirname(path)) + "_" + os.path.basename(path))
    with Image.open(path) as im:
        im = im.convert("RGB")
        bg = Image.new("RGB", im.size, (255, 255, 255))
        diff = ImageChops.difference(im, bg)
        # tolerate near-white anti-aliasing
        diff = ImageChops.add(diff, diff, 2.0, -20)
        bbox = diff.getbbox()
        if bbox is None:
            im.save(out)
            return out
        l, t, r, b = bbox
        l, t = max(0, l - pad), max(0, t - pad)
        r, b = min(im.width, r + pad), min(im.height, b + pad)
        im = im.crop((l, t, r, b))
        # limit resolution (about 260 dpi at the 6.2 in print width) and store as JPEG
        # so the document stays a reasonable size
        if im.width > MAX_FIG_PX:
            im = im.resize((MAX_FIG_PX, round(im.height * MAX_FIG_PX / im.width)), Image.LANCZOS)
        out = os.path.splitext(out)[0] + ".jpg"
        im.save(out, "JPEG", quality=85, optimize=True)
    return out


# --------------------------------------------------------------------------- main
def add_experiment(d, exp_dir, content):
    n = content["number"]
    d.add_paragraph(f"Experiment No. {n}", style="LabExpNo")
    d.add_paragraph(content["title"], style="LabExpTitle")

    info_table(d, [
        ("Experiment No.", str(n)),
        ("Title", content["title"]),
        ("Date of experiment", ""),
        ("Tools / software used", content.get("tools", "")),
        ("Input data", content.get("dataset", "")),
        ("Program file", os.path.basename(content.get("code_file", ""))),
    ])

    heading2(d, "1. Aim")
    body(d, content["aim"])

    if content.get("objectives"):
        heading2(d, "2. Objectives")
        for obj in content["objectives"]:
            bullet(d, obj)

    heading2(d, "3. Theory")
    for item in content["theory"]:
        if item.startswith("## "):
            heading3(d, item[3:].strip())
        else:
            body_or_equation(d, item)

    heading2(d, "4. Algorithm / Procedure")
    for i, step in enumerate(content["algorithm"], 1):
        step = step.strip()
        # strip a leading "Step k:" or "k." if the author already numbered it
        for prefix in (f"Step {i}:", f"Step {i}.", f"Step {i} -", f"{i}.", f"{i})"):
            if step.lower().startswith(prefix.lower()):
                step = step[len(prefix):].strip()
                break
        numbered(d, i, step)

    heading2(d, "5. Output")
    for fig in content["figures"]:
        fpath = os.path.join(exp_dir, fig["file"])
        if not os.path.exists(fpath):
            body(d, f"[missing figure {fig['file']}]")
            continue
        figure(d, fpath, fig["caption"])

    console = clean_console(read_text(os.path.join(exp_dir, "console.txt")))
    if console:
        heading3(d, "Console output")
        code_block(d, console, style="LabConsole", fill="F7F7F7", max_lines=MAX_CONSOLE_LINES)
        d.add_paragraph(style="LabSmall")

    heading2(d, "6. Results and Observations")
    for item in content["results"]:
        body_or_equation(d, item)

    heading2(d, "7. Conclusion")
    body_or_equation(d, content["conclusion"])


def setup_header_footer(d):
    sec = d.sections[0]
    sec.different_first_page_header_footer = True
    # first page (cover/index) keeps empty header and footer
    header = sec.header
    header.is_linked_to_previous = False
    hp = header.paragraphs[0]
    hp.text = ""
    r = hp.add_run("Computer Vision (CSE3010)  |  Lab File  |  Fall Semester July 2026")
    r.font.size = Pt(9)
    r.font.color.rgb = GREY
    r.font.name = BODY_FONT
    hp.alignment = WD_ALIGN_PARAGRAPH.RIGHT
    footer = sec.footer
    footer.is_linked_to_previous = False
    fp = footer.paragraphs[0]
    fp.text = ""
    fp.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = fp.add_run("Page ")
    r.font.size = Pt(9)
    r.font.name = BODY_FONT
    r2 = fp.add_run()
    r2.font.size = Pt(9)
    r2.font.name = BODY_FONT
    add_field(r2, "PAGE")


def fix_index_typos(d):
    """The supplied index truncates 'Surveillance'; repair it in place (run level, keeps formatting)."""
    fixes = {"Surveillanc": "Surveillance"}
    for table in d.tables:
        for row in table.rows:
            for cell in row.cells:
                for p in cell.paragraphs:
                    for r in p.runs:
                        for bad, good in fixes.items():
                            if bad in r.text and good not in r.text:
                                r.text = r.text.replace(bad, good)


def main():
    d = docx.Document(TEMPLATE)
    make_styles(d)
    setup_header_footer(d)
    fix_index_typos(d)

    # remove the trailing empty paragraph after the index table so the first
    # experiment's page break starts cleanly
    last = d.paragraphs[-1] if d.paragraphs else None
    if last is not None and not last.text.strip():
        last._p.getparent().remove(last._p)

    exp_dirs = sorted(dd for dd in os.listdir(CONTENT_ROOT) if dd.startswith("exp"))
    built, missing = [], []
    for dd in exp_dirs:
        exp_dir = os.path.join(CONTENT_ROOT, dd)
        cpath = os.path.join(exp_dir, "content.json")
        if not os.path.exists(cpath):
            missing.append(dd)
            continue
        with open(cpath, "r", encoding="utf-8") as f:
            content = json.load(f)
        add_experiment(d, exp_dir, content)
        built.append(content["number"])

    d.save(OUT_PATH)
    print("saved", OUT_PATH)
    print("experiments built:", built)
    if missing:
        print("WARNING no content.json for:", missing)


if __name__ == "__main__":
    main()
