import os
import time
import json
import uuid
from pathlib import Path
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv
from tqdm.auto import tqdm
from pinecone import Pinecone, ServerlessSpec
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document
import asyncio
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

class AdvancedVectorStore:
    """Advanced document processing and vector storage system"""
    
    def __init__(self):
        self.setup_components()
        self.upload_dir = Path("./uploaded_docs")
        self.upload_dir.mkdir(exist_ok=True)
        
    def setup_components(self):
        """Initialize all components"""
        try:
            # Pinecone setup
            self.pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
            self.index_name = os.getenv("PINECONE_INDEX_NAME")
            self.pinecone_env = os.getenv("PINECONE_ENV")
            
            # Create index if it doesn't exist
            self.ensure_index_exists()
            self.index = self.pc.Index(self.index_name)
            
            # Initialize models
            self.embed_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
            self.llm = ChatGroq(
                temperature=0,
                model_name="llama3-8b-8192",
                groq_api_key=os.getenv("GROQ_API_KEY")
            )
            
            # Setup prompts
            self.setup_prompts()
            
            logger.info("✅ Advanced Vector Store initialized successfully")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize Advanced Vector Store: {e}")
            raise
    
    def setup_prompts(self):
        """Setup prompt templates for document processing"""
        
        # Document analysis prompt
        self.doc_analysis_prompt = PromptTemplate.from_template("""
Analyze this document excerpt and extract structured information:

Document Type: {doc_type}
Content: {content}

Extract and return in JSON format:
{{
    "main_topics": ["topic1", "topic2", ...],
    "medical_entities": ["entity1", "entity2", ...],
    "key_concepts": ["concept1", "concept2", ...],
    "relationships": [["entity1", "relationship", "entity2"], ...],
    "summary": "Brief summary of the content",
    "importance_score": 0.0-1.0
}}

JSON:
""")
        
        # Content summarization prompt
        self.summary_prompt = PromptTemplate.from_template("""
Create a concise, informative summary of this medical document content:

Content: {content}

Provide a 2-3 sentence summary that captures the key medical information:
""")
    
    def ensure_index_exists(self):
        """Ensure Pinecone index exists"""
        try:
            existing_indexes = [i["name"] for i in self.pc.list_indexes()]
            
            if self.index_name not in existing_indexes:
                logger.info(f"Creating new Pinecone index: {self.index_name}")
                
                self.pc.create_index(
                    name=self.index_name,
                    dimension=768,  # Google embedding dimension
                    metric="cosine",
                    spec=ServerlessSpec(
                        cloud='aws',
                        region=self.pinecone_env
                    )
                )
                
                # Wait for index to be ready
                while not self.pc.describe_index(self.index_name).status["ready"]:
                    logger.info("Waiting for index to be ready...")
                    time.sleep(2)
                    
                logger.info("✅ Pinecone index created successfully")
            else:
                logger.info(f"✅ Using existing Pinecone index: {self.index_name}")
                
        except Exception as e:
            logger.error(f"Failed to ensure index exists: {e}")
            raise
    
    async def analyze_document_content(self, content: str, doc_type: str) -> Dict[str, Any]:
        """Analyze document content and extract structured information"""
        try:
            # Limit content size for analysis
            analysis_content = content[:2000] if len(content) > 2000 else content
            
            response = await asyncio.to_thread(
                self.llm.invoke,
                self.doc_analysis_prompt.format(
                    doc_type=doc_type,
                    content=analysis_content
                )
            )
            
            # Parse JSON response
            try:
                analysis = json.loads(response.content)
                return analysis
            except json.JSONDecodeError:
                # Fallback if JSON parsing fails
                return {
                    "main_topics": [],
                    "medical_entities": [],
                    "key_concepts": [],
                    "relationships": [],
                    "summary": content[:200] + "...",
                    "importance_score": 0.5
                }
                
        except Exception as e:
            logger.error(f"Document analysis failed: {e}")
            return {
                "main_topics": [],
                "medical_entities": [],
                "key_concepts": [],
                "relationships": [],
                "summary": "Analysis unavailable",
                "importance_score": 0.5
            }
    
    async def generate_content_summary(self, content: str) -> str:
        """Generate a concise summary of content"""
        try:
            summary_content = content[:1500] if len(content) > 1500 else content
            
            response = await asyncio.to_thread(
                self.llm.invoke,
                self.summary_prompt.format(content=summary_content)
            )
            
            return response.content.strip()
            
        except Exception as e:
            logger.error(f"Summary generation failed: {e}")
            return content[:150] + "..." if len(content) > 150 else content
    
    def create_multiple_chunk_strategies(self, documents: List[Document]) -> List[Dict]:
        """Create chunks using multiple strategies - FIXED DOCUMENT ACCESS"""
        
        all_chunks = []
        
        # Strategy 1: Standard recursive splitting
        standard_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50,
            separators=["\n\n", "\n", ". ", " "]
        )
        standard_chunks = standard_splitter.split_documents(documents)
        
        for i, chunk in enumerate(standard_chunks):
            all_chunks.append({
                "content": chunk.page_content,  # FIXED: Use attribute access
                "metadata": {
                    **chunk.metadata,  # FIXED: Use attribute access
                    "chunk_id": f"std_{i}",
                    "chunk_strategy": "standard",
                    "chunk_size": len(chunk.page_content)
                }
            })
        
        # Strategy 2: Larger chunks for context
        large_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=100,
            separators=["\n\n", "\n", ". "]
        )
        large_chunks = large_splitter.split_documents(documents)
        
        for i, chunk in enumerate(large_chunks):
            all_chunks.append({
                "content": chunk.page_content,  # FIXED: Use attribute access
                "metadata": {
                    **chunk.metadata,  # FIXED: Use attribute access
                    "chunk_id": f"lg_{i}",
                    "chunk_strategy": "large_context",
                    "chunk_size": len(chunk.page_content)
                }
            })
        
        # Strategy 3: Small chunks for precise retrieval
        small_splitter = RecursiveCharacterTextSplitter(
            chunk_size=250,
            chunk_overlap=25,
            separators=[". ", "\n", " "]
        )
        small_chunks = small_splitter.split_documents(documents)
        
        for i, chunk in enumerate(small_chunks):
            if len(chunk.page_content.strip()) > 50:  # Skip very small chunks
                all_chunks.append({
                    "content": chunk.page_content,  # FIXED: Use attribute access
                    "metadata": {
                        **chunk.metadata,  # FIXED: Use attribute access
                        "chunk_id": f"sm_{i}",
                        "chunk_strategy": "precise",
                        "chunk_size": len(chunk.page_content)
                    }
                })
        
        return all_chunks
    
    async def process_and_store_documents(self, uploaded_files: List, role: str, 
                                         doc_id: str, doc_type: str = "medical",
                                         priority: str = "medium") -> Dict[str, Any]:
        """Enhanced document processing with multiple techniques - FIXED"""
        
        processing_stats = {
            "files_processed": 0,
            "total_chunks": 0,
            "embeddings_created": 0,
            "analysis_completed": 0,
            "errors": []
        }
        
        try:
            for file in uploaded_files:
                logger.info(f"Processing file: {file.filename}")
                
                # Save file
                file_path = self.upload_dir / f"{doc_id}_{file.filename}"
                
                # Handle different file input types
                if hasattr(file, 'read'):  # UploadFile object
                    content = await file.read()
                    with open(file_path, "wb") as f:
                        f.write(content)
                else:  # File-like object
                    with open(file_path, "wb") as f:
                        f.write(file.file.read())
                
                # Load document
                loader = PyPDFLoader(str(file_path))
                documents = loader.load()
                
                # Create multiple chunking strategies
                all_chunks = self.create_multiple_chunk_strategies(documents)
                processing_stats["total_chunks"] += len(all_chunks)
                
                # Process chunks in batches
                batch_size = 10
                for i in range(0, len(all_chunks), batch_size):
                    batch = all_chunks[i:i + batch_size]
                    await self.process_chunk_batch(
                        batch, file.filename, role, doc_id, doc_type, priority
                    )
                    processing_stats["embeddings_created"] += len(batch)
                
                # Analyze document for insights
                full_content = "\n".join([doc.page_content for doc in documents])
                analysis = await self.analyze_document_content(full_content[:3000], doc_type)
                processing_stats["analysis_completed"] += 1
                
                # Store document metadata
                await self.store_document_metadata(
                    file.filename, doc_id, role, doc_type, priority, analysis
                )
                
                processing_stats["files_processed"] += 1
                logger.info(f"✅ Completed processing: {file.filename}")
                
        except Exception as e:
            error_msg = f"Document processing error: {str(e)}"
            logger.error(error_msg)
            processing_stats["errors"].append(error_msg)
        
        return processing_stats
    
    async def process_chunk_batch(self, chunks: List[Dict], filename: str, 
                                 role: str, doc_id: str, doc_type: str, priority: str):
        """Process a batch of chunks - FIXED PINECONE UPSERT"""
        
        try:
            # Generate embeddings for all chunks in batch
            texts = [chunk["content"] for chunk in chunks]
            embeddings = await asyncio.to_thread(
                self.embed_model.embed_documents, texts
            )
            
            # Generate summaries for chunks
            summaries = []
            for chunk in chunks:
                summary = await self.generate_content_summary(chunk["content"])
                summaries.append(summary)
            
            # Prepare vectors for upsert - FIXED FORMAT
            vectors_to_upsert = []
            
            for i, (chunk, embedding, summary) in enumerate(zip(chunks, embeddings, summaries)):
                vector_id = f"{doc_id}_{chunk['metadata']['chunk_id']}_{i}"
                
                # Comprehensive metadata
                metadata = {
                    "text": chunk["content"],
                    "summary": summary,
                    "source": filename,
                    "doc_id": doc_id,
                    "role": role,
                    "doc_type": doc_type,
                    "priority": priority,
                    "chunk_strategy": chunk["metadata"]["chunk_strategy"],
                    "chunk_size": chunk["metadata"]["chunk_size"],
                    "page": chunk["metadata"].get("page", 0),
                    "processed_at": datetime.utcnow().isoformat(),
                    "vector_type": "full_content"
                }
                
                # FIXED: Use proper Pinecone vector format (id, values, metadata)
                vectors_to_upsert.append({
                    "id": vector_id,
                    "values": embedding,
                    "metadata": metadata
                })
                
                # Also create a summary vector for alternative retrieval
                if len(summary) > 50:  # Only if summary is substantial
                    summary_embedding = await asyncio.to_thread(
                        self.embed_model.embed_query, summary
                    )
                    
                    summary_metadata = metadata.copy()
                    summary_metadata.update({
                        "text": summary,
                        "vector_type": "summary",
                        "original_text_preview": chunk["content"][:200]
                    })
                    
                    vectors_to_upsert.append({
                        "id": f"{vector_id}_summary",
                        "values": summary_embedding,
                        "metadata": summary_metadata
                    })
            
            # Upsert to Pinecone - FIXED FORMAT
            if vectors_to_upsert:
                await asyncio.to_thread(self.index.upsert, vectors=vectors_to_upsert)
                logger.info(f"✅ Upserted {len(vectors_to_upsert)} vectors for {filename}")
            
        except Exception as e:
            logger.error(f"Batch processing failed: {e}")
            raise
    
    async def store_document_metadata(self, filename: str, doc_id: str, 
                                    role: str, doc_type: str, priority: str, 
                                    analysis: Dict[str, Any]):
        """Store document metadata in database"""
        try:
            from config.db import db_manager
            
            metadata = {
                "filename": filename,
                "doc_id": doc_id,
                "role": role,
                "doc_type": doc_type,
                "priority": priority,
                "accessible_roles": [role, "admin"],  # Admin can access all
                "upload_timestamp": datetime.utcnow(),
                "analysis": analysis,
                "status": "processed"
            }
            
            await db_manager.store_document_metadata(metadata)
            logger.info(f"✅ Stored metadata for document: {filename}")
            
        except Exception as e:
            logger.error(f"Failed to store document metadata: {e}")
    
    async def search_similar_documents(self, query: str, role: str, 
                                     doc_type: Optional[str] = None,
                                     top_k: int = 5) -> List[Dict]:
        """Search for similar documents"""
        try:
            # Generate query embedding
            query_embedding = await asyncio.to_thread(
                self.embed_model.embed_query, query
            )
            
            # Build filter
            filter_dict = {"role": role}
            if doc_type:
                filter_dict["doc_type"] = doc_type
            
            # Search
            results = await asyncio.to_thread(
                self.index.query,
                vector=query_embedding,
                top_k=top_k,
                include_metadata=True,
                filter=filter_dict
            )
            
            # Format results
            formatted_results = []
            for match in results.get("matches", []):
                formatted_results.append({
                    "id": match["id"],
                    "score": match["score"],
                    "content": match["metadata"].get("text", ""),
                    "source": match["metadata"].get("source", ""),
                    "summary": match["metadata"].get("summary", ""),
                    "doc_type": match["metadata"].get("doc_type", ""),
                    "metadata": match["metadata"]
                })
            
            return formatted_results
            
        except Exception as e:
            logger.error(f"Similar document search failed: {e}")
            return []
    
    def get_index_stats(self) -> Dict[str, Any]:
        """Get index statistics - FIXED JSON serialization"""
        try:
            stats = self.index.describe_index_stats()
            
            # Safely convert to JSON-serializable format
            safe_stats = {
                "total_vectors": getattr(stats, 'total_vector_count', 0),
                "dimension": getattr(stats, 'dimension', 768),
                "index_fullness": getattr(stats, 'index_fullness', 0.0),
                "namespaces": {}
            }
            
            # Handle namespaces safely
            if hasattr(stats, 'namespaces') and stats.namespaces:
                for name, namespace_stats in stats.namespaces.items():
                    safe_stats["namespaces"][name] = {
                        "vector_count": getattr(namespace_stats, 'vector_count', 0)
                    }
            
            return safe_stats
            
        except Exception as e:
            logger.error(f"Failed to get index stats: {e}")
            return {
                "total_vectors": 0,
                "dimension": 768,
                "index_fullness": 0.0,
                "namespaces": {},
                "error": str(e)
            }

# Global instance
advanced_vectorstore = AdvancedVectorStore()
