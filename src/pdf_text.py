import os
import re
from pathlib import Path
import pdfplumber

RAW_PDF_DIR = Path("data/raw/rbi_texts/statements/")      
OUTPUT_DIR = Path("data/text/rbi_policy")        


def extract_date_from_filename(filename: str) -> str:
    patterns = [r"(\d{4})[-_]?(\d{2})[-_]?(\d{2})"]   # 2014-04-01 or 2014_04_01 or 20140401

    for pattern in patterns:
        match = re.search(pattern, filename)
        if match:
            y, m, d = match.groups()
            return f"{y}-{m}-{d}"
    return None


def clean_text(text: str) -> str:

    text = re.sub(r"\n\s*\n", "\n\n", text)
    text = re.sub(r"^\s*\d+\s*$", "", text, flags=re.MULTILINE)
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


def extract_text_from_pdf(pdf_path: Path) -> str:
    #Extract readable text from a PDF using pdfplumber
    all_text = []
    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages):
            text = page.extract_text()

            if text:
                all_text.append(text)
    return "\n".join(all_text)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    pdf_files = list(RAW_PDF_DIR.glob("*.pdf"))

    if not pdf_files:
        raise FileNotFoundError(
            f"No PDFs found in {RAW_PDF_DIR}. ")

    print(f"Found {len(pdf_files)} PDFs. Starting extraction...\n")

    for pdf_path in pdf_files:
        print(f"Processing: {pdf_path.name}")
        date_str = extract_date_from_filename(pdf_path.name)

        if date_str is None:
            date_str = pdf_path.stem
            print(f"  ⚠️ Could not auto-detect date — using '{date_str}' as identifier.")

        out_file = OUTPUT_DIR / f"{date_str}_policy.txt"

        raw_text = extract_text_from_pdf(pdf_path)
        cleaned_text = clean_text(raw_text)

        with open(out_file, "w", encoding="utf-8") as f:
            f.write(cleaned_text)

        print(f"  → Saved to: {out_file}")

    print("\n✅ All PDFs converted successfully!")

if __name__ == "__main__":
    main()