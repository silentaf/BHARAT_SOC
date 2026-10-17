"""Build the submission PDF of docs/technical_report.md (Markdown -> HTML -> PDF via headless Chrome).

    python scripts/build_report.py            # -> outputs/report/QC-Assist_PS5_Phase1_report.pdf

Needs the `markdown` package (requirements.txt) and Google Chrome or Chromium.
"""

import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import markdown

ROOT = Path(__file__).resolve().parent.parent

CSS = """
@page { size: A4; margin: 16mm 15mm; }
body { font: 10.5pt/1.45 "Noto Sans", "DejaVu Sans", Arial, sans-serif; color: #111; }
h1 { font-size: 19pt; margin: 0 0 4pt; } h2 { font-size: 14pt; margin: 16pt 0 6pt; border-bottom: 1px solid #ccc; }
h3 { font-size: 11.5pt; margin: 12pt 0 4pt; }
h1, h2, h3 { break-after: avoid; page-break-after: avoid; }
table { border-collapse: collapse; margin: 6pt 0; font-size: 9pt; width: 100%; page-break-inside: avoid; }
th, td { border: 1px solid #bbb; padding: 3pt 5pt; text-align: left; vertical-align: top; }
th { background: #f0f0f0; }
code { font: 9pt "DejaVu Sans Mono", monospace; background: #f4f4f4; padding: 0 2pt; }
pre { background: #f4f4f4; padding: 6pt; font-size: 7.5pt; white-space: pre; overflow: hidden; page-break-inside: avoid; }
pre code { background: none; padding: 0; font-size: 7.5pt; }
blockquote { margin: 6pt 0; padding: 4pt 10pt; border-left: 3px solid #2f6fdf; background: #f5f8ff; }
img { max-width: 100%; border: 1px solid #ccc; }
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=str(ROOT / "docs" / "technical_report.md"))
    ap.add_argument("--out", default=str(ROOT / "outputs" / "report" / "QC-Assist_PS5_Phase1_report.pdf"))
    args = ap.parse_args()

    chrome = next((shutil.which(c) for c in ("google-chrome", "chromium", "chromium-browser") if shutil.which(c)), None)
    if not chrome:
        sys.exit("needs Google Chrome or Chromium for the PDF step")
    src, out = Path(args.src), Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    body = markdown.markdown(src.read_text(), extensions=["tables", "fenced_code", "sane_lists"])
    html = out.with_suffix(".html")
    # <base> makes relative image paths (images/...) resolve next to the Markdown file
    html.write_text(f'<!doctype html><html><head><meta charset="utf-8"><base href="{src.parent.as_uri()}/">'
                    f"<style>{CSS}</style></head><body>{body}</body></html>")
    with tempfile.TemporaryDirectory(dir=out.parent) as profile:  # throwaway Chrome profile
        subprocess.run([chrome, "--headless=new", "--no-first-run", "--disable-gpu", f"--user-data-dir={profile}",
                        "--no-pdf-header-footer", f"--print-to-pdf={out}", html.as_uri()],
                       check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=120)
    print(f"{out} ({out.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
