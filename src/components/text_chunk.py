from src.logging.logger import logging
from langchain.text_splitter import RecursiveCharacterTextSplitter


class MetadataChunker:
    def __init__(self, chunk_size=1000, chunk_overlap=200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=["\n\n", "\n", " "]
        )

    def build_metadata_tuples(self, sections, cleaned_pages, offset=0, source=None, doc_type=None):
        adjusted_sections = [(name, start_page - offset) for name, start_page in sections]
        sections_sorted = sorted(adjusted_sections, key=lambda x: x[1])
        max_page = max(page['page_num'] for page in cleaned_pages)
        
        section_ranges = []
        for idx, (section_name, start_page) in enumerate(sections_sorted):
            end_page = sections_sorted[idx + 1][1] - 1 if idx < len(sections_sorted) - 1 else max_page
            section_ranges.append((section_name, start_page, end_page))

        metadata_pages = []
        for page in cleaned_pages:
            page_num = page['page_num']
            assigned_section = next(
                (name for name, start, end in section_ranges if start <= page_num <= end),
                "Not Assigned"
            )
            page_with_metadata = page.copy()
            page_with_metadata['section'] = assigned_section
            if source:
                page_with_metadata['source'] = source
            if doc_type:
                page_with_metadata['doc_type'] = doc_type
            metadata_pages.append(page_with_metadata)

        logging.info(f"Annotated {len(metadata_pages)} pages with section metadata.")
        return metadata_pages

    def langchain_chunk_texts(self, metadata_pages):
        chunks = []
        for page in metadata_pages:
            splits = self.splitter.split_text(page['text'])
            for chunk in splits:
                chunks.append({
                    "content": chunk,
                    "section": page['section'],
                    "page_nums": page['page_num'],
                    "source": page["source"],
                    "doc_type": page["doc_type"]
                })
        logging.info(f"Generated {len(chunks)} text chunks.")
        return chunks

    def convert_chunks_to_documents(self, chunks):
        docs = [{
            "context": chunk["content"],
            "metadata": {
                "section": chunk["section"],
                "page_num": chunk["page_nums"],
                "source": chunk["source"],
                "doc_type": chunk["doc_type"]
            }
        } for chunk in chunks]
        logging.info(f"Converted {len(docs)} chunks to document format.")
        return docs

