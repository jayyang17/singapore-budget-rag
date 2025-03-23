import os
import re
import logging
from PyPDF2 import PdfReader
from typing import List, Tuple, Dict
from src.logging.logger import logging

class PDFParser:
    def __init__(self, pdf_path: str):
        if not os.path.exists(pdf_path):
            logging.error(f"PDF not found at path: {pdf_path}")
            raise FileNotFoundError(f"PDF not found at path: {pdf_path}")
        self.pdf_path = pdf_path
        self.reader = PdfReader(pdf_path)
        self.filename = os.path.basename(pdf_path)
        logging.info(f"Loaded PDF: {self.filename}")

    def extract_sections_from_toc(self, toc_pages: Tuple[int, int] = (0, 1)) -> List[Tuple[str, int]]:
        logging.info(f"Extracting TOC sections from pages {toc_pages}")
        sections = []
        for i in range(toc_pages[0], toc_pages[1] + 1):
            try:
                page = self.reader.pages[i]
                text = page.extract_text()
                for line in text.split('\n'):
                    match = re.match(r'^([A-Z]\.\s.+?)\.{3,}\s+(\d+)$', line.strip())
                    if match:
                        title = match.group(1).strip()
                        page_num = int(match.group(2))
                        sections.append((title, page_num))
            except Exception as e:
                logging.warning(f"Failed to process TOC page {i}: {e}")
        logging.info(f"Extracted {len(sections)} section entries from TOC")
        return sections

    @staticmethod
    def clean_section_titles(raw_sections: List[Tuple[str, int]]) -> List[Tuple[str, int]]:
        logging.info(f"Cleaning {len(raw_sections)} section titles")
        cleaned = []
        for raw_title, page_num in raw_sections:
            title = re.sub(r'^[A-Z]\.\s+', '', raw_title)
            title = re.sub(r'\.{2,}', '', title)
            title = re.sub(r'\s{2,}', ' ', title)
            cleaned.append((title.strip(), page_num))
        logging.info("Section titles cleaned")
        return cleaned

    def extract_page_texts(self, start_page: int = 0, page_offset: int = 0) -> List[Dict[str, str]]:
        logging.info(f"Extracting page texts from page {start_page} onward")
        pages = []
        for page_num in range(start_page, len(self.reader.pages)):
            try:
                raw_text = self.reader.pages[page_num].extract_text()
                if not raw_text:
                    logging.debug(f"Page {page_num} is empty. Skipping.")
                    continue
                pages.append({
                    "page_num": page_num + page_offset,
                    "text": self.clean_text(raw_text)
                })
            except Exception as e:
                logging.warning(f"Failed to extract text from page {page_num}: {e}")
        logging.info(f"Extracted text from {len(pages)} pages")
        return pages

    @staticmethod
    def clean_text(text: str) -> str:
        cleaned_lines = []
        for line in text.split('\n'):
            if re.search(r'\bPage\s*\d+\s+of\s+\d+\b', line, re.IGNORECASE):
                continue
            cleaned_lines.append(line)
        return '\n'.join(cleaned_lines).strip()

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Extract and clean TOC and text from a Singapore Budget PDF.")
    parser.add_argument("pdf_path", type=str, help="Path to the budget PDF file.")
    parser.add_argument("--start_page", type=int, default=2, help="Page to start extracting text from (default: 2).")
    parser.add_argument("--page_offset", type=int, default=1, help="Offset to apply to logical page numbers (default: 1).")
    args = parser.parse_args()

    try:
        logging.info(f"Processing PDF: {args.pdf_path}")
        pdf_parser = PDFBudgetParser(args.pdf_path)

        # Extract and clean sections from ToC
        raw_sections = pdf_parser.extract_sections_from_toc()
        cleaned_sections = pdf_parser.clean_section_titles(raw_sections)
        logging.info("Cleaned TOC Sections:")
        for title, page in cleaned_sections:
            print(f"- {title} (Page {page})")

        print("\n" + "=" * 60 + "\n")

        # Extract and clean page texts
        cleaned_pages = pdf_parser.extract_page_texts(start_page=args.start_page, page_offset=args.page_offset)
        logging.info(f"Sample cleaned page (Page {cleaned_pages[0]['page_num']}):\n")
        print(cleaned_pages[0]["text"][:1000])  # Show first 1000 characters

    except Exception as e:
        logging.error(f"An error occurred: {e}")

# python -m src.components.data_ingest knowledge_base/2024/fy2024_budget_statement.pdf --start_page 2 --page_offset 1

