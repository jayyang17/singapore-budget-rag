import os
import glob
from dotenv import load_dotenv

from src.logging.logger import logging
from src.components.chunk_pipeline import ChunkPipeline
from src.components.embeddings import Embedder
from src.components.prompts import ChatAssistant
from langchain.chat_models import ChatOpenAI
from openai import OpenAI

from langchain.vectorstores import Chroma
import gradio as gr

# ========= CONFIG VARIABLES ========= #
PDF_DIR = "knowledge_base"
DB_NAME = "budget_vector_db"
openai = OpenAI()
OPENAI_MODEL = "gpt-4o"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
TEMPERATURE = 0.7

# ========= SYSTEM MESSAGES CONFIG ========= #
DEFAULT_CONFIG = {
    "retriever_k": 25,
    "system_prompt": (
        "You are a highly reliable assistant answering questions based on Singapore’s FY2024 and FY2025 budget documents. "
        "You MUST use only the content from these documents. You MAY summarize and synthesize across them, including comparing statistics across FY2024 and FY2025 when both are available in the retrieved content. Do not guess or fabricate any data. "
        "If the answer is not found within the provided documents, respond: 'I do not have sufficient information based on the provided documents.' "
        "Only mention fiscal years FY2024 and FY2025 unless another year is explicitly stated in the documents. "
        "If ask to compare fiscal years performance, mention a Note that comparison across years are not yet optimzed"
        "If a value is mentioned in the documents, and the documents are for the requested fiscal year, provide that value, even if the year is not directly stated next to the value."
    )
}

EMBEDDINGS_CLASS = Embedder
# ==================================== #
load_dotenv(override=True)
api_key = os.getenv('OPENAI_API_KEY')

# Ingest & chunk
pdf_files = glob.glob(f"{PDF_DIR}/**/*.pdf", recursive=True)

if not pdf_files:
    logging.warning("No PDF files found.")
    exit()

rag_pipeline = ChunkPipeline(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
all_chunks = []

for pdf_path in pdf_files:
    logging.info(f"Processing: {pdf_path}")
    chunks = rag_pipeline.process_pdf(pdf_path)
    all_chunks.extend(chunks)

# Embed
embedder = EMBEDDINGS_CLASS(db_name=DB_NAME)
vectorstore = embedder.run(all_chunks)

# Initialize Chat Assistant
llm = ChatOpenAI(temperature=TEMPERATURE, model_name=OPENAI_MODEL)
chat_assistant = ChatAssistant(llm=llm, vectorstore=vectorstore, config=DEFAULT_CONFIG)

# Gradio App
def ask_question(user_input, history=[]):
    return chat_assistant.chat(user_input, history)

gr.ChatInterface(fn=ask_question, title="Singapore Budget RAG", theme="default").launch(share=True)
