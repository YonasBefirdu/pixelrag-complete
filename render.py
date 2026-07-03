"""
Step 1 — Render PDF pages to image tiles.
Usage: python render.py path/to/report.pdf

Uses pymupdf (fitz) for PDFs — no browser, 10-50x faster than pixelshot.
Falls back to pixelshot for URLs.

Why pymupdf instead of pixelshot for PDFs:
  pixelshot uses Playwright (headless Chromium) which is designed for web pages.
  For PDFs it launches a browser per page, which is extremely slow at scale.
  pymupdf renders PDF pages directly to pixels — no browser, no network, just math.
"""

import sys
import subprocess
from pathlib import Path


DPI = 150  # 150 DPI -> ~1240x1754px for A4 (good quality, reasonable file size)
           # 200 DPI -> ~1654x2339px (matches paper benchmark resolution, larger files)


def render_pdf(pdf_path: str):
    try:
        import pymupdf as fitz
    except ImportError:
        try:
            import fitz
        except ImportError:
            print("[ERROR] pymupdf not installed. Run: pip install pymupdf")
            sys.exit(1)

    pdf = Path(pdf_path).resolve()
    if not pdf.exists():
        print(f"[ERROR] File not found: {pdf}")
        sys.exit(1)

    output_dir = Path("tiles")
    output_dir.mkdir(exist_ok=True)

    doc = fitz.open(str(pdf))
    total = len(doc)
    print(f"[1/1] Rendering {pdf.name} ({total} pages) at {DPI} DPI -> {output_dir}/")

    mat = fitz.Matrix(DPI / 72, DPI / 72)  # 72 is PDF default DPI

    for i, page in enumerate(doc):
        pix = page.get_pixmap(matrix=mat)
        out_path = output_dir / f"page_{i+1:05d}.png"
        pix.save(str(out_path))

        if (i + 1) % 50 == 0 or (i + 1) == total:
            print(f"  {i+1}/{total} pages rendered")

    doc.close()
    tiles = list(output_dir.glob("*.png"))
    print(f"[DONE] {len(tiles)} tiles saved to ./{output_dir}/")


def render_url(url: str):
    output_dir = Path("tiles")
    output_dir.mkdir(exist_ok=True)

    print(f"[1/1] Rendering {url} -> {output_dir}/ (using pixelshot)")
    result = subprocess.run(
        ["pixelshot", url, "--output", str(output_dir)],
        capture_output=False,
    )
    if result.returncode != 0:
        print("[ERROR] pixelshot failed. Is pixelrag installed? Run: pip install pixelrag")
        sys.exit(1)

    tiles = list(output_dir.glob("*.png")) + list(output_dir.glob("*.jpg"))
    print(f"[DONE] {len(tiles)} tiles saved to ./{output_dir}/")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python render.py path/to/report.pdf")
        print("       python render.py https://example.com")
        sys.exit(1)

    target = sys.argv[1]
    if target.startswith("http://") or target.startswith("https://"):
        render_url(target)
    else:
        render_pdf(target)
