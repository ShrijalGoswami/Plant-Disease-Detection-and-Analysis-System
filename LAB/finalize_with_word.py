"""
Finalise the lab file with Microsoft Word (COM automation):
  1. open the built docx, read the page number of every experiment heading,
  2. write those page numbers into the index table ("Page no." column),
  3. update fields, save, and export a PDF,
  4. render the PDF pages to PNG thumbnails for visual checking.

Usage: python finalize_with_word.py <in.docx> <out.docx> <out.pdf> <png_dir>
"""
import os
import sys
import json

import win32com.client as win32

WD_ACTIVE_END_PAGE_NUMBER = 3
WD_EXPORT_FORMAT_PDF = 17

in_docx, out_docx, out_pdf, png_dir = [os.path.abspath(a) for a in sys.argv[1:5]]
os.makedirs(png_dir, exist_ok=True)

word = win32.gencache.EnsureDispatch("Word.Application")
word.Visible = False
word.DisplayAlerts = 0
doc = word.Documents.Open(in_docx)
try:
    doc.Repaginate()
    # collect page numbers for experiment headings (style LabExpNo)
    pages = {}
    for para in doc.Paragraphs:
        try:
            style_name = para.Style.NameLocal
        except Exception:
            continue
        if style_name == "LabExpNo":
            text = para.Range.Text.strip()
            num = int(text.replace("Experiment No.", "").strip())
            pages[num] = para.Range.Information(WD_ACTIVE_END_PAGE_NUMBER)
    print("experiment start pages:", json.dumps(pages))

    # fill the index table: rows 4..17 (1-based) hold experiments 1..14, column 4 = Page no.
    table = doc.Tables(1)
    filled = 0
    for r in range(1, table.Rows.Count + 1):
        try:
            first = table.Cell(r, 1).Range.Text.strip().strip("\r\x07").strip()
        except Exception:
            continue
        if first.isdigit() and int(first) in pages:
            cell = table.Cell(r, 4)
            cell.Range.Text = str(pages[int(first)])
            cell.Range.ParagraphFormat.Alignment = 1  # centre
            filled += 1
    print("index rows filled:", filled)

    doc.Fields.Update()
    total_pages = doc.ComputeStatistics(2)  # wdStatisticPages
    print("total pages:", total_pages)
    doc.SaveAs2(out_docx, FileFormat=16)  # wdFormatDocumentDefault (docx)
    doc.ExportAsFixedFormat(out_pdf, WD_EXPORT_FORMAT_PDF)
    print("saved", out_docx, "and", out_pdf)
finally:
    doc.Close(False)
    word.Quit()

import fitz  # pymupdf
pdf = fitz.open(out_pdf)
print("pdf pages:", pdf.page_count)
for i, page in enumerate(pdf):
    pix = page.get_pixmap(dpi=60)
    pix.save(os.path.join(png_dir, f"page_{i + 1:03d}.png"))
print("rendered", pdf.page_count, "pages to", png_dir)
