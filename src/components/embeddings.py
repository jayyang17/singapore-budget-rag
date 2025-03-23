import os
from typing import List
from src.logging.logger import logging

from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain_core.documents import Document

class Embedder:
    def __init__(self, db_name: str = "budget_vector_db"):
        self.db_name = db_name
        self.embeddings = OpenAIEmbeddings()
        logging.info("Embedder initialized with OpenAI embeddings.")

    def convert_to_documents(self, chunks: List[dict]) -> List[Document]:
        logging.info("Converting chunks to LangChain Document objects...")
        return [
            Document(page_content=chunk["context"], metadata=chunk["metadata"])
            for chunk in chunks
        ]

    def create_vectorstore(self, documents: List[Document]):
        if os.path.exists(self.db_name):
            logging.warning(f"Existing DB found at '{self.db_name}', deleting collection...")
            Chroma(persist_directory=self.db_name, embedding_function=self.embeddings).delete_collection()

        logging.info("Creating Chroma vectorstore...")
        vectorstore = Chroma.from_documents(
            documents=documents,
            embedding=self.embeddings,
            persist_directory=self.db_name
        )

        count = vectorstore._collection.count()
        logging.info(f"Vectorstore created with {count} documents.")
        return vectorstore

    def run(self, chunks: List[dict]):
        documents = self.convert_to_documents(chunks)
        return self.create_vectorstore(documents)
