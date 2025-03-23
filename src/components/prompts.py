import re
from src.logging.logger import logging
from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
import re

DEFAULT_CONFIG = {
    "retriever_k": 25,
    "system_prompt": (
        "You are a highly reliable assistant answering questions based on Singapore’s FY2024 and FY2025 budget documents. "
        "You MUST use only the content from these documents. You MAY summarize and synthesize across them, including comparing statistics across FY2024 and FY2025 when both are available in the retrieved content. Do not guess or fabricate any data. "
        "If the answer is not even indirectly supported by the documents, respond: 'I do not have sufficient information based on the provided documents.' "
        "Only mention fiscal years FY2024 and FY2025 unless another year is explicitly stated in the documents."
    )
}

class ChatAssistant:
    def __init__(self, llm, vectorstore, config: dict = DEFAULT_CONFIG):
        self.llm = llm
        self.vectorstore = vectorstore
        self.config = config

        self.prompt = ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template(config["system_prompt"]),
            HumanMessagePromptTemplate.from_template(
                "Here are some documents to help you:\n\n{context}\n\n"
                "Now answer the question: {question}\n\n"
                "If the documents do not directly answer the question, but contain related figures or closely associated statistics, you MAY include them with a clear disclaimer explaining how they are related."
            )
        ])

        self.memory = ConversationBufferMemory(
            memory_key='chat_history',
            output_key='answer',
            return_messages=True
        )

    def _detect_filter(self, question: str):
        match = re.search(r'fy\s*(20\d{2})|(?:[^0-9]|^)(20\d{2})(?:[^0-9]|$)', question.lower())
        year = match.group(1) or match.group(2) if match else None

        if "compare" in question.lower() and "2024" in question and "2025" in question:
            return {}
        elif year in {"2024", "2025"}:
            return {"where": {"source": {"$eq": f"fy{year}_budget_statement.pdf"}}}
        return {}

    def _get_retriever(self, question: str):
        filter_kwargs = self._detect_filter(question)
        logging.info(f"[Chat] Retrieval filter: {filter_kwargs.get('where', 'None')}")

        return self.vectorstore.as_retriever(search_kwargs={
            "k": self.config.get("retriever_k", 25),
            **({"filter": filter_kwargs["where"]} if "where" in filter_kwargs else {})
        })

    def chat(self, question: str, history=None) -> str:
        retriever = self._get_retriever(question)
        docs = retriever.invoke(question)

        # Retry with no filter
        if not docs:
            logging.info("[Chat] No docs found. Retrying without filter...")
            retriever = self.vectorstore.as_retriever(search_kwargs={"k": self.config.get("retriever_k", 25)})
            docs = retriever.invoke(question)

        # Setup chain
        chain = ConversationalRetrievalChain.from_llm(
            llm=self.llm,
            retriever=retriever,
            memory=self.memory,
            return_source_documents=True,
            combine_docs_chain_kwargs={"prompt": self.prompt},
            output_key="answer"
        )

        result = chain.invoke({"question": question})
        answer = result.get("answer", "")
        source_docs = result.get("source_documents", [])

        citations = []
        for i, doc in enumerate(source_docs, 1):
            md = doc.metadata
            citations.append(f"**Doc {i}:** {md.get('source')} | {md.get('section')} | Page {md.get('page_num')}")

        return answer + ("\n\n**Sources:**\n" + "\n".join(citations) if citations else "\n\n**No Sources Found**")
