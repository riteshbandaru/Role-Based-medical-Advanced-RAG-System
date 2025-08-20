from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import os

from auth.routes import router as auth_router
from docs.routes import router as docs_router
from chat.routes import router as chat_router
from graph.routes import router as graph_router

app = FastAPI(
    title="Advanced Healthcare RBAC RAG System",
    description="Multi-vector hybrid retrieval with graph-based knowledge system",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create upload directory
os.makedirs("uploaded_docs", exist_ok=True)

# Mount static files
app.mount("/static", StaticFiles(directory="uploaded_docs"), name="static")

# Include routers
app.include_router(auth_router, prefix="/auth", tags=["Authentication"])
app.include_router(docs_router, prefix="/docs", tags=["Document Management"])
app.include_router(chat_router, prefix="/chat", tags=["Chat & Retrieval"])
app.include_router(graph_router, prefix="/graph", tags=["Knowledge Graph"])

@app.get("/")
async def root():
    return {
        "message": "Advanced Healthcare RAG System API",
        "version": "2.0.0",
        "features": [
            "Multi-vector Hybrid Retrieval",
            "Query Rewriting & Routing",
            "Multi-hop Chain-of-Thought",
            "Graph-based Retrieval",
            "Role-based Access Control"
        ]
    }

@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "timestamp": "2025-08-20T15:21:00Z",
        "components": {
            "api": "operational",
            "database": "operational",
            "vector_store": "operational",
            "knowledge_graph": "operational"
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
