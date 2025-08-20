import os
import asyncio
from dotenv import load_dotenv
from pinecone import Pinecone
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
from typing import Dict, List
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

# Environment variables
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")

# Set environment variable for Google API
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY

class SimpleRAG:
    def __init__(self):
        self.setup_components()
    
    def setup_components(self):
        """Initialize RAG components"""
        try:
            # Initialize Pinecone
            self.pc = Pinecone(api_key=PINECONE_API_KEY)
            self.index = self.pc.Index(PINECONE_INDEX_NAME)
            
            # Initialize embedding model
            self.embed_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
            
            # Initialize LLM
            self.llm = ChatGroq(
                temperature=0.3,
                model_name="llama3-8b-8192",
                groq_api_key=GROQ_API_KEY
            )
            
            # Simple RAG prompt
            self.rag_prompt = PromptTemplate.from_template("""
You are a helpful healthcare assistant. Answer the following question based only on the provided context.

Question: {question}
Context: {context}

Instructions:
- Provide accurate, helpful medical information
- If you cannot answer based on the context, say so clearly
- Include relevant document sources if available
- Be professional and empathetic

Answer:
""")
            
            logger.info("✅ Simple RAG components initialized successfully")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize Simple RAG components: {e}")
            raise

    async def simple_query(self, query: str, user_role: str) -> Dict:
        """Execute simple RAG query"""
        try:
            # Generate embedding for query
            embedding = await asyncio.to_thread(
                self.embed_model.embed_query, query
            )
            
            # Search in Pinecone
            search_results = await asyncio.to_thread(
                self.index.query,
                vector=embedding,
                top_k=5,
                include_metadata=True,
                filter={"role": user_role}
            )
            
            # Extract context and sources
            contexts = []
            sources = set()
            
            for match in search_results.get("matches", []):
                metadata = match.get("metadata", {})
                text = metadata.get("text", "")
                source = metadata.get("source", "Unknown")
                
                if text:
                    contexts.append(text)
                    sources.add(source)
            
            if not contexts:
                return {
                    "answer": "I couldn't find relevant information to answer your question based on the available documents.",
                    "sources": [],
                    "method": "Simple RAG",
                    "context_found": False
                }
            
            # Combine contexts
            combined_context = "\n\n".join(contexts[:3])  # Limit to top 3 for performance
            
            # Generate answer
            response = await asyncio.to_thread(
                self.llm.invoke,
                self.rag_prompt.format(question=query, context=combined_context)
            )
            
            return {
                "answer": response.content,
                "sources": list(sources),
                "method": "Simple RAG - Vector Similarity Search",
                "context_found": True,
                "num_sources": len(sources)
            }
            
        except Exception as e:
            logger.error(f"Simple RAG query failed: {e}")
            return {
                "answer": f"I encountered an error while processing your question: {str(e)}",
                "sources": [],
                "method": "Simple RAG",
                "context_found": False,
                "error": str(e)
            }

# Global instance
simple_rag = SimpleRAG()

async def answer_query(query: str, user_role: str) -> Dict:
    """Legacy function for backwards compatibility"""
    return await simple_rag.simple_query(query, user_role)
