"""OCR an image-only policy PDF into searchable PDF and UTF-8 sidecar text."""

from __future__ import annotations

import argparse
import os
import subprocess
import tempfile
from pathlib import Path

import fitz
from pypdf import PdfReader, PdfWriter


DEFAULT_TESSERACT = Path(r"C:\Program Files\Tesseract-OCR\tesseract.exe")


def run_ocr(
    input_pdf: Path,
    output_pdf: Path,
    output_text: Path,
    *,
    language: str = "vie+eng",
    dpi: int = 300,
    tesseract_path: Path = DEFAULT_TESSERACT,
) -> None:
    if not tesseract_path.exists():
        raise FileNotFoundError(f"Tesseract not found: {tesseract_path}")

    output_pdf.parent.mkdir(parents=True, exist_ok=True)
    output_text.parent.mkdir(parents=True, exist_ok=True)

    document = fitz.open(input_pdf)
    all_text: list[str] = []

    with tempfile.TemporaryDirectory(prefix="policy-ocr-") as temp_dir:
        temp_path = Path(temp_dir)
        page_pdfs: list[Path] = []

        for page_index, page in enumerate(document, start=1):
            image_path = temp_path / f"page-{page_index:03d}.png"
            output_base = temp_path / f"page-{page_index:03d}"
            matrix = fitz.Matrix(dpi / 72, dpi / 72)
            page.get_pixmap(matrix=matrix, alpha=False).save(image_path)

            env = os.environ.copy()
            env["TESSDATA_PREFIX"] = str(tesseract_path.parent / "tessdata")
            subprocess.run(
                [
                    str(tesseract_path),
                    str(image_path),
                    str(output_base),
                    "-l",
                    language,
                    "--psm",
                    "3",
                    "pdf",
                    "txt",
                ],
                check=True,
                env=env,
                capture_output=True,
                text=True,
            )

            page_pdf = output_base.with_suffix(".pdf")
            page_text = output_base.with_suffix(".txt")
            page_pdfs.append(page_pdf)
            text = page_text.read_text(encoding="utf-8", errors="replace").strip()
            all_text.append(f"--- Trang {page_index} ---\n{text}")

        writer = PdfWriter()
        for page_pdf in page_pdfs:
            reader = PdfReader(str(page_pdf))
            for pdf_page in reader.pages:
                writer.add_page(pdf_page)
        with output_pdf.open("wb") as file:
            writer.write(file)

    document.close()
    output_text.write_text("\n\n".join(all_text) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input_pdf", type=Path)
    parser.add_argument("output_pdf", type=Path)
    parser.add_argument("output_text", type=Path)
    parser.add_argument("--language", default="vie+eng")
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--tesseract", type=Path, default=DEFAULT_TESSERACT)
    args = parser.parse_args()
    run_ocr(
        args.input_pdf,
        args.output_pdf,
        args.output_text,
        language=args.language,
        dpi=args.dpi,
        tesseract_path=args.tesseract,
    )


if __name__ == "__main__":
    main()
