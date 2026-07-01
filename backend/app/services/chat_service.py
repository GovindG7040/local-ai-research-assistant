import os

from langchain_core.output_parsers import StrOutputParser

from app.prompts.chat_prompt import chat_prompt
from app.services.memory_service import MemoryService
from app.services.ollama_service import OllamaService
from app.services.reranker_service import RerankerService
from app.vectorstore.chroma_manager import ChromaManager
from app.prompts.query_rewrite_prompt import query_rewrite_prompt
from app.services.bm25_instance import bm25_service

# =====================================================
# FORMAT RETRIEVED DOCUMENTS
# =====================================================

def format_docs(documents):
    """
    Converts retrieved documents into a single context string.
    """

    return "\n\n".join(

        document.page_content

        for document in documents

    )


# =====================================================
# CHAT SERVICE
# =====================================================

class ChatService:
    """
    Handles the complete Conversational RAG pipeline.

    Flow:

    Question
        ↓
    Retriever
        ↓
    CrossEncoder Reranker
        ↓
    Prompt
        ↓
    Ollama
        ↓
    Sources
    """

    def __init__(self):

        # ----------------------------------------
        # Vector Store
        # ----------------------------------------

        self.chroma_manager = ChromaManager()

        self.vector_store = (
            self.chroma_manager.get_vector_store()
        )

        # ----------------------------------------
        # Retriever
        # ----------------------------------------

        self.retriever = (

            self.vector_store.as_retriever(

                search_kwargs={

                    "k": 20,

                }

            )

        )

        # ----------------------------------------
        # Reranker
        # ----------------------------------------

        self.reranker = RerankerService()

        # ----------------------------------------
        # Conversation Memory
        # ----------------------------------------

        self.memory = MemoryService()

        # ----------------------------------------
        # LLM
        # ----------------------------------------

        self.ollama_service = OllamaService()

        self.llm = (
            self.ollama_service.get_llm()
        )

        # ----------------------------------------
        # LCEL Chain
        # ----------------------------------------

        # ----------------------------------------
        # Query Rewrite Chain
        # ----------------------------------------

        self.query_rewrite_chain = (

            query_rewrite_prompt

            | self.llm

            | StrOutputParser()

        )

        self.chain = (

            chat_prompt

            | self.llm

            | StrOutputParser()

        )

    # =====================================================
    # ASK QUESTION
    # =====================================================

    def ask(self, question: str):

        # ----------------------------------------
        # Session ID
        # ----------------------------------------

        session_id = "default"

        # ----------------------------------------
        # Get Previous Conversation
        # ----------------------------------------

        chat_history = self.memory.get_history(
            session_id
        )

        # ----------------------------------------
        # Rewrite Query using Conversation History
        # ----------------------------------------

        if len(chat_history) == 0:

            retrieval_query = question

        else:

            retrieval_query = self.query_rewrite_chain.invoke(

                {

                    "chat_history": chat_history,

                    "question": question,

                }

            )

        print("\n========== HISTORY-AWARE RETRIEVAL ==========")
        print(f"Original Question : {question}")
        print(f"Retrieval Query   : {retrieval_query}")

        # ----------------------------------------
        # Retrieve Documents
        # ----------------------------------------

        # ----------------------------------------
        # Dense Retrieval (Chroma)
        # ----------------------------------------

        dense_documents = self.retriever.invoke(
            retrieval_query
        )

        # ----------------------------------------
        # Keyword Retrieval (BM25)
        # ----------------------------------------

        bm25_documents = bm25_service.retrieve(

            query=retrieval_query,

            k=20,

        )

        # ----------------------------------------
        # Merge Retrieval Results
        # ----------------------------------------

        documents = dense_documents + bm25_documents

        # ----------------------------------------
        # Remove Duplicate Chunks
        # ----------------------------------------

        unique_documents = []

        seen = set()

        for document in documents:

            key = (

                document.metadata.get("source"),

                document.page_content,

            )

            if key not in seen:

                seen.add(key)

                unique_documents.append(document)

        documents = unique_documents

        print("\n========== HYBRID SEARCH ==========")
        print(f"Dense Results : {len(dense_documents)}")
        print(f"BM25 Results  : {len(bm25_documents)}")
        print(f"Merged Results: {len(documents)}")

        # ----------------------------------------
        # Rerank Documents
        # ----------------------------------------

        documents = self.reranker.rerank(

            question=retrieval_query,

            documents=documents,

            top_k=5,

        )

        # ----------------------------------------
        # Build Context
        # ----------------------------------------

        context = format_docs(
            documents
        )

        # ----------------------------------------
        # Generate Answer
        # ----------------------------------------

        answer = self.chain.invoke(

            {

                "chat_history": chat_history,

                "context": context,

                "question": question,

            }

        )

        # ----------------------------------------
        # Store Conversation
        # ----------------------------------------

        self.memory.add_user_message(

            session_id,

            question,

        )

        self.memory.add_ai_message(

            session_id,

            answer,

        )

        # ----------------------------------------
        # Extract Source Files
        # ----------------------------------------

        sources = []

        seen = set()

        for document in documents:

            source = document.metadata.get(
                "source"
            )

            if source:

                filename = os.path.basename(
                    source
                )

                if filename not in seen:

                    seen.add(filename)

                    sources.append(

                        {

                            "file": filename

                        }

                    )

        # ----------------------------------------
        # Return Response
        # ----------------------------------------

        return {

            "answer": answer,

            "sources": sources,

        }