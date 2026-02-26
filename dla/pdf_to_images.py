import os
import glob
from pathlib import Path
from pdf2image import convert_from_path
from tqdm import tqdm
import argparse


def save_pdf_pages(pdf_path: Path, pdf_pages_dir: Path, base_name: str) -> None:
    """Extract each page as a separate PDF file using pypdf."""
    try:
        from pypdf import PdfReader, PdfWriter
    except ImportError:
        print("Warning: pypdf not installed. Skipping PDF page extraction. pip install pypdf")
        return
    reader = PdfReader(str(pdf_path))
    for i in range(len(reader.pages)):
        page_num = i + 1
        out_name = f"{base_name}_page_{page_num:03d}.pdf"
        out_path = pdf_pages_dir / out_name
        writer = PdfWriter()
        writer.add_page(reader.pages[i])
        with open(out_path, "wb") as f:
            writer.write(f)


def main():
    parser = argparse.ArgumentParser(description="Convert PDFs to images")
    parser.add_argument("--input_dir", type=str, required=True, help="Directory containing PDF files")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save images (or base dir if --save_pdf_pages)")
    parser.add_argument("--dpi", type=int, default=300, help="DPI for conversion")
    parser.add_argument("--fmt", type=str, default="png", help="Output image format")
    parser.add_argument("--filter", type=str, default=None,
                        help="Keyword filter for PDF filenames (e.g. 'problem' to select only *_problem.pdf)")
    parser.add_argument("--files", nargs="*", default=None,
                        help="Optional list of PDF filenames to process (e.g. sat-practice-test-5.pdf)")
    parser.add_argument("--save_pdf_pages", action="store_true",
                        help="Also save each page as a separate PDF in output_dir/pdf_pages (requires pypdf)")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    pdf_files = sorted(input_dir.glob("*.pdf"))
    if args.files:
        names_set = set(args.files)
        pdf_files = [f for f in pdf_files if f.name in names_set]
        print(f"Filtered to specified files: {len(pdf_files)} PDF files.")
    elif args.filter:
        pdf_files = [f for f in pdf_files if args.filter in f.stem]
        print(f"Filtered with keyword '{args.filter}': {len(pdf_files)} PDF files.")
    if not pdf_files:
        print(f"No PDF files found in {input_dir}")
        return

    print(f"Found {len(pdf_files)} PDF files.")

    if args.save_pdf_pages:
        png_dir = output_dir / "png_pages"
        pdf_dir = output_dir / "pdf_pages"
        png_dir.mkdir(parents=True, exist_ok=True)
        pdf_dir.mkdir(parents=True, exist_ok=True)
        image_out_dir = png_dir
    else:
        output_dir.mkdir(parents=True, exist_ok=True)
        image_out_dir = output_dir

    for pdf_path in tqdm(pdf_files, desc="Processing PDFs"):
        try:
            images = convert_from_path(str(pdf_path), dpi=args.dpi)
            base_name = pdf_path.stem
            for i, image in enumerate(images):
                page_num = i + 1
                image_name = f"{base_name}_page_{page_num:03d}.{args.fmt}"
                image_path = image_out_dir / image_name
                image.save(image_path, args.fmt.upper())
            if args.save_pdf_pages:
                save_pdf_pages(pdf_path, pdf_dir, base_name)
        except Exception as e:
            print(f"Error processing {pdf_path}: {e}")
            print("Make sure 'poppler' is installed and in your PATH.")

if __name__ == "__main__":
    main()
