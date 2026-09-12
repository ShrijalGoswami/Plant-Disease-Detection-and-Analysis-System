"""
Build the VITyarthi project report PDF from docs/report.md.

Usage (from the project root):
    python docs/build_report.py --name "Student Name" --reg "21BCE0000" --faculty "Faculty Name" --date "12 September 2026"

Steps:
1. docs/report.md is converted to HTML (python-markdown).
2. Images become numbered figures with captions, tables get their captions
   from the "Table N:" paragraph written just above them.
3. A cover page and a table of contents are generated.
4. Headless Chromium (Playwright) prints the body to PDF with a running
   header and "Page x of y" footer. A first pass is used only to find on
   which page every section starts, so the table of contents can show
   page numbers. The cover is printed separately without header or footer
   and the two PDFs are merged with pypdf.

Output: docs/Project_Report.pdf
"""

import argparse
import base64
import html
import io
import re
from pathlib import Path

import markdown
from PIL import Image
from playwright.sync_api import sync_playwright
from pypdf import PdfReader, PdfWriter

DOCS = Path(__file__).resolve().parent
SOURCE = DOCS / "report.md"
OUTPUT = DOCS / "Project_Report.pdf"
TITLE = "Plant Disease Detection and Analysis System"
SUBTITLE = "Leaf image classification with a fine tuned MobileNetV2"

CSS = """
@page { size: A4; margin: 22mm 20mm 24mm 20mm; }
@page wide { size: A4 landscape; margin: 18mm 15mm 20mm 15mm; }
.landscape { page: wide; page-break-before: always; page-break-after: always; }
.landscape figure img { max-width: 100%; max-height: 165mm; }
html { font-size: 11pt; }
body { font-family: Cambria, "Times New Roman", serif; color: #1a1a1a; line-height: 1.45; margin: 0; }
h1 { font-size: 17pt; margin: 0 0 10pt 0; padding-bottom: 4pt; border-bottom: 1px solid #333; page-break-after: avoid; }
h1.section { page-break-before: always; }
h2 { font-size: 13pt; margin: 16pt 0 6pt 0; page-break-after: avoid; }
p { margin: 0 0 8pt 0; text-align: justify; }
ul, ol { margin: 0 0 8pt 0; padding-left: 20pt; }
li { margin-bottom: 3pt; }
code { font-family: Consolas, monospace; font-size: 9.5pt; background: #f3f3f3; padding: 0 2px; }
pre { font-family: Consolas, monospace; font-size: 9pt; background: #f5f5f5; border: 1px solid #ddd; padding: 8pt; white-space: pre-wrap; page-break-inside: avoid; }
pre code { background: none; padding: 0; }
table { border-collapse: collapse; width: 100%; margin: 4pt 0 12pt 0; font-size: 9.5pt; page-break-inside: auto; }
th, td { border: 1px solid #999; padding: 4pt 6pt; vertical-align: top; text-align: left; }
th { background: #ececec; font-weight: bold; }
tr { page-break-inside: avoid; }
p.caption { font-size: 9.5pt; font-style: italic; text-align: center; margin: 4pt 0 10pt 0; }
p.table-caption { font-size: 9.5pt; font-style: italic; text-align: left; margin: 10pt 0 2pt 0; page-break-after: avoid; }
figure { margin: 10pt 0 12pt 0; text-align: center; page-break-inside: avoid; }
figure img { max-width: 100%; max-height: 190mm; border: 1px solid #ccc; }
figure.tall img { max-height: 215mm; }
figcaption { font-size: 9.5pt; font-style: italic; margin-top: 4pt; text-align: center; }
.toc h1 { border-bottom: 1px solid #333; }
.toc table { font-size: 11pt; }
.toc td { border: none; padding: 3pt 4pt; }
.toc td.num { text-align: right; width: 10%; }
.toc tr.sub td:first-child { padding-left: 18pt; font-size: 10pt; }
.cover { text-align: center; font-family: Cambria, serif; }
.cover .inst { font-size: 13pt; margin-top: 40mm; letter-spacing: 1px; }
.cover .title { font-size: 26pt; font-weight: bold; margin-top: 45mm; line-height: 1.25; }
.cover .subtitle { font-size: 13pt; margin-top: 8mm; color: #333; }
.cover .kind { font-size: 12pt; margin-top: 30mm; }
.cover table { width: 70%; margin: 18mm auto 0 auto; font-size: 12pt; border: none; }
.cover td { border: none; padding: 4pt 8pt; text-align: left; }
.cover td.k { font-weight: bold; width: 45%; }
"""


def load_figure(path: Path, trim_blank_bottom: bool) -> tuple[str, float]:
    """
    Return the image as a data URI plus its width/height ratio.

    Screenshots of the app were taken with a tall browser window, so some
    end in a long run of identical blank rows. Those rows are trimmed here
    for the report only; the files in assets/ are left untouched.
    """
    image = Image.open(path).convert("RGB")
    if trim_blank_bottom:
        pixels = image.load()
        width, height = image.size
        last_row = [pixels[x, height - 1] for x in range(0, width, 4)]
        bottom = height - 1
        while bottom > 0 and [pixels[x, bottom - 1] for x in range(0, width, 4)] == last_row:
            bottom -= 1
        image = image.crop((0, 0, width, min(height, bottom + 60)))
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    uri = "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()
    return uri, image.size[0] / image.size[1]


def convert_body(md_text: str) -> tuple[str, list[tuple[int, str, str]]]:
    """Markdown -> HTML with figures, table captions and heading ids."""
    body = markdown.markdown(md_text, extensions=["tables", "fenced_code"])

    # Images -> figures. The alt text holds the caption.
    def figure(match):
        alt, src = html.unescape(match.group(1)), match.group(2)
        path = (DOCS / src).resolve()
        is_screenshot = "screenshots" in src
        uri, ratio = load_figure(path, trim_blank_bottom=is_screenshot)
        tall = is_screenshot or "confusion_matrix" in src
        fig = (f'<figure class="{"tall" if tall else ""}"><img src="{uri}" alt="{html.escape(alt)}">'
               f"<figcaption>{html.escape(alt)}</figcaption></figure>")
        # Very wide diagrams are unreadable in portrait, so they get a landscape page.
        return f'<div class="landscape">{fig}</div>' if ratio > 2.2 and not is_screenshot else fig
    body = re.sub(r'<p><img alt="([^"]*)" src="([^"]+)"\s*/?></p>', figure, body)

    # "Table N: ..." paragraphs become table captions.
    body = re.sub(r"<p>(Table \d+: .*?)</p>", r'<p class="table-caption">\1</p>', body)

    # Give headings ids and collect them for the table of contents.
    headings = []

    def heading(match):
        level, text = int(match.group(1)), match.group(2)
        anchor = f"sec-{len(headings)}"
        headings.append((level, re.sub(r"<[^>]+>", "", text), anchor))
        cls = ' class="section"' if level == 1 else ""
        return f'<h{level} id="{anchor}"{cls}>{text}</h{level}>'
    body = re.sub(r"<h([12])>(.*?)</h\1>", heading, body)
    return body, headings


def cover_html(args) -> str:
    rows = [("Student name", args.name), ("Registration number", args.reg),
            ("Faculty", args.faculty), ("Date of submission", args.date)]
    table = "".join(f'<tr><td class="k">{html.escape(k)}</td><td>{html.escape(v)}</td></tr>' for k, v in rows)
    return f"""<html><head><meta charset="utf-8"><style>{CSS}</style></head><body class="cover">
<div class="inst">VITyarthi<br>Build Your Own Project</div>
<div class="title">{TITLE}</div>
<div class="subtitle">{SUBTITLE}</div>
<div class="kind">Project Report<br>Course: Computer Vision</div>
<table>{table}</table>
</body></html>"""


def toc_html(headings, pages: dict[str, int] | None) -> str:
    rows = []
    for level, text, anchor in headings:
        page = pages.get(anchor, "") if pages else ""
        cls = ' class="sub"' if level == 2 else ""
        rows.append(f"<tr{cls}><td>{html.escape(text)}</td><td class=\"num\">{page}</td></tr>")
    return f'<div class="toc"><h1>Contents</h1><table>{"".join(rows)}</table></div>'


def body_html(body: str, headings, pages=None) -> str:
    return f"""<html><head><meta charset="utf-8"><style>{CSS}</style></head><body>
{toc_html(headings, pages)}
{body}
</body></html>"""


HEADER = f'<div style="font-family:Cambria,serif;font-size:8.5pt;color:#555;width:100%;padding:0 20mm;">{TITLE}</div>'
FOOTER = ('<div style="font-family:Cambria,serif;font-size:8.5pt;color:#555;width:100%;padding:0 20mm;'
          'text-align:center;">Page <span class="pageNumber"></span> of <span class="totalPages"></span></div>')


def print_pdf(page, content: str, path: Path, header_footer: bool) -> None:
    page.set_content(content, wait_until="load")
    page.pdf(path=str(path), format="A4", print_background=True, prefer_css_page_size=True,
             display_header_footer=header_footer, header_template=HEADER if header_footer else "<span></span>",
             footer_template=FOOTER if header_footer else "<span></span>")


def find_section_pages(pdf_path: Path, headings) -> dict[str, int]:
    """Locate the first body page on which each heading text appears."""
    reader = PdfReader(str(pdf_path))
    pages = {}
    texts = [" ".join((p.extract_text() or "").split()) for p in reader.pages]
    # The contents list can run over more than one page and it repeats every
    # heading, so searching must start at the first body page: the first page
    # whose text begins with the first section heading (every level 1 heading
    # starts a new page).
    first_heading = " ".join(headings[0][1].split())
    first_body = next(i for i, t in enumerate(texts) if t.startswith(first_heading))
    for level, text, anchor in headings:
        needle = " ".join(text.split())
        for index, page_text in enumerate(texts[first_body:], start=first_body + 1):
            if needle in page_text:
                pages[anchor] = index
                break
    return pages


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the project report PDF")
    parser.add_argument("--name", required=True)
    parser.add_argument("--reg", required=True)
    parser.add_argument("--faculty", required=True)
    parser.add_argument("--date", required=True)
    args = parser.parse_args()

    body, headings = convert_body(SOURCE.read_text(encoding="utf-8"))
    tmp_dir = DOCS / "_build"
    tmp_dir.mkdir(exist_ok=True)
    cover_pdf, body_pdf = tmp_dir / "cover.pdf", tmp_dir / "body.pdf"

    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page()
        print_pdf(page, cover_html(args), cover_pdf, header_footer=False)
        # First pass without page numbers, only to measure where sections land.
        print_pdf(page, body_html(body, headings), body_pdf, header_footer=True)
        pages = find_section_pages(body_pdf, headings)
        print_pdf(page, body_html(body, headings, pages), body_pdf, header_footer=True)
        browser.close()

    writer = PdfWriter()
    for part in (cover_pdf, body_pdf):
        for pdf_page in PdfReader(str(part)).pages:
            writer.add_page(pdf_page)
    writer.add_metadata({"/Title": TITLE, "/Author": args.name})
    with open(OUTPUT, "wb") as f:
        writer.write(f)
    total = len(PdfReader(str(OUTPUT)).pages)
    print(f"Wrote {OUTPUT} ({total} pages)")


if __name__ == "__main__":
    main()
