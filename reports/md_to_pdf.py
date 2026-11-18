"""Render a Markdown file to PDF (Markdown -> styled HTML -> headless Chromium print).
Usage: python reports/md_to_pdf.py reports/PROGRESS_REPORT_AND_VIVA_GUIDE.md
"""
import sys
from pathlib import Path

import markdown
from playwright.sync_api import sync_playwright

src = Path(sys.argv[1]).resolve()
out = src.with_suffix(".pdf")
body = markdown.markdown(src.read_text(encoding="utf-8"), extensions=["tables", "fenced_code", "sane_lists"])
css = """
body{font-family:'Segoe UI',Arial,sans-serif;font-size:10pt;line-height:1.45;color:#1b2533;margin:0}
h1{color:#16365c;font-size:20pt;border-bottom:3px solid #16365c;padding-bottom:4px}
h2{color:#16365c;font-size:14pt;margin-top:22px;border-bottom:1px solid #c9d3e0;padding-bottom:2px;page-break-after:avoid}
h3{color:#16365c;font-size:11.5pt;margin-top:14px;page-break-after:avoid}
table{border-collapse:collapse;width:100%;margin:8px 0;font-size:8.6pt}
tr{page-break-inside:avoid}thead{display:table-header-group}
th{background:#dce6f2;text-align:left}th,td{border:1px solid #b8c2cf;padding:3px 5px;vertical-align:top}
tr:nth-child(even) td{background:#f6f8fb}
code{font-family:Consolas,monospace;font-size:8.8pt;background:#eef2f7;padding:0 3px;border-radius:3px}
hr{border:0;border-top:1px solid #c9d3e0;margin:16px 0}li{margin:2px 0}em{color:#4a5568}
"""
html = f"<!doctype html><html><head><meta charset='utf-8'><style>{css}</style></head><body>{body}</body></html>"
with sync_playwright() as p:
    b = p.chromium.launch()
    pg = b.new_page()
    pg.set_content(html, wait_until="load")
    pg.pdf(path=str(out), format="A4", print_background=True,
           margin={"top": "16mm", "bottom": "16mm", "left": "14mm", "right": "14mm"},
           display_header_footer=True, header_template="<span></span>",
           footer_template="<div style='font-size:8px;width:100%;text-align:center;color:#777'>PathSnap – Progress Report &amp; Viva Guide · page <span class='pageNumber'></span>/<span class='totalPages'></span></div>")
    b.close()
print(out)
