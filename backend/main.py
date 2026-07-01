from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.ingest import router as ingest_router
from app.api.chat import router as chat_router
from contextlib import asynccontextmanager

from app.services.ingest_service import IngestService
from app.services.bm25_instance import bm25_service

# =====================================================
# APPLICATION LIFESPAN
# =====================================================

@asynccontextmanager
async def lifespan(app: FastAPI):

    print("\n========== BUILDING BM25 INDEX ==========\n")

    ingest_service = IngestService()

    chunks = ingest_service.load_chunks()

    bm25_service.build_index(chunks)

    print("\n========== BM25 READY ==========\n")

    yield

app = FastAPI(
    title="Local AI Research Assistant",
    description="A local RAG application built using FastAPI, LangChain, Ollama, and ChromaDB.",
    version="1.0.0",
    lifespan=lifespan
)

# =====================================================
# CORS
# =====================================================

app.add_middleware(
    CORSMiddleware,

    allow_origins=[
        "http://localhost:5173",
    ],

    allow_credentials=True,

    allow_methods=["*"],

    allow_headers=["*"],
)


app.include_router(ingest_router)
app.include_router(chat_router)


@app.get("/")
def root():
    return {
        "message": "Welcome to the Local AI Research Assistant API!"
    }

@app.get("/health")
def health_check():
    return {
        "status": "healthy"
    }