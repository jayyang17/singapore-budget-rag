import os
from typing import List, Dict

from src.logging.logger import logging
from src.components.data_ingest import PDFParser
from src.components.text_chunk import MetadataChunker

class ChunkPipeline:
    def __init__(self, chunk_size=1000, chunk_overlap=200):
        self.chunker = MetadataChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    def process_pdf(self, pdf_path: str, toc_pages=(0, 1), start_page=2, page_offset=1) -> List[Dict]:
        logging.info(f"Processing: {pdf_path}")

        parser = PDFParser(pdf_path)
        raw_sections = parser.extract_sections_from_toc(toc_pages)
        cleaned_sections = parser.clean_section_titles(raw_sections)
        cleaned_pages = parser.extract_page_texts(start_page=start_page, page_offset=page_offset)

        filename = os.path.basename(pdf_path)
        year_match = "2024" if "fy2024" in filename.lower() else "2025" if "fy2025" in filename.lower() else None
        doc_type = f"budget_statement_{year_match}" if year_match else "budget_statement"

        metadata_pages = self.chunker.build_metadata_tuples(
            cleaned_sections, cleaned_pages,
            offset=page_offset, source=filename, doc_type=doc_type
        )

        chunks = self.chunker.langchain_chunk_texts(metadata_pages)
        docs = self.chunker.convert_chunks_to_documents(chunks)

        return docs
