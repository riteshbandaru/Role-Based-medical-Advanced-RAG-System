import os
import asyncio
import json
from typing import List, Dict, Any, Tuple, Optional
from datetime import datetime
from dotenv import load_dotenv
from pinecone import Pinecone
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain_core.documents import Document
import networkx as nx
from rank_bm25 import BM25Okapi
import numpy as np
import logging
import time

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

class AdvancedRAGEngine:
    """Advanced RAG engine with multi-vector hybrid retrieval and knowledge graph"""
    
    def __init__(self):
        self.setup_components()
        self.knowledge_graph = nx.DiGraph()
        self.bm25_docs_cache = {}  # Cache for BM25 documents
        self.last_cache_update = 0
        self.cache_ttl = 300  # 5 minutes cache TTL
        
    def setup_components(self):
        """Initialize all RAG components"""
        try:
            # Pinecone setup
            self.pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
            self.index = self.pc.Index(os.getenv("PINECONE_INDEX_NAME"))
            
            # Embedding model
            self.embed_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
            
            # LLM
            self.llm = ChatGroq(
                temperature=0.3,
                model_name="llama3-8b-8192",
                groq_api_key=os.getenv("GROQ_API_KEY")
            )
            
            # Advanced prompts
            self.setup_prompts()
            
            logger.info("✅ Advanced RAG Engine initialized successfully")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize Advanced RAG Engine: {e}")
            raise
    
    def setup_prompts(self):
        """Setup all prompt templates"""
        
        # Query rewriting prompt
        self.query_rewriter_prompt = PromptTemplate.from_template("""
You are a medical query optimization specialist. Rewrite the user's query to be more specific and effective for medical document retrieval.

User Role: {role}
Original Query: {query}

Guidelines:
- For doctors: Use precise medical terminology, include relevant specialties
- For nurses: Focus on care procedures, protocols, and patient management
- For patients: Simplify language, include context for better understanding
- For admin: Include system-wide and management perspectives

Rewritten Query (return only the improved query):
""")
        
        # Multi-hop reasoning prompt
        self.multi_hop_prompt = PromptTemplate.from_template("""
You are performing step-by-step medical information retrieval.

Original Query: {query}
Current Step: {step}
Step Description: {step_description}

Retrieved Information:
{context}

Based on this information, provide:
1. Summary of findings for this step
2. Whether additional information is needed (YES/NO)
3. If YES, provide the next specific search query

Format your response as:
SUMMARY: [your summary]
NEEDS_MORE: [YES/NO]
NEXT_QUERY: [next search query if needed]
""")
        
        # Entity extraction prompt
        self.entity_prompt = PromptTemplate.from_template("""
Extract medical entities from this text. Focus on:
- Diseases and conditions
- Symptoms
- Medications and treatments
- Medical procedures
- Body parts/anatomy
- Medical professionals/specialties

Text: {text}

Return only the entities as a comma-separated list:
""")
        
        # Final answer generation prompt
        self.final_answer_prompt = PromptTemplate.from_template("""
You are a knowledgeable healthcare assistant providing comprehensive answers.

Original Query: {query}
User Role: {role}

Retrieved Context:
{context}

Knowledge Graph Relationships:
{graph_context}

Multi-step Analysis:
{multihop_summary}

Provide a comprehensive, accurate answer that:
1. Directly addresses the user's question
2. Uses appropriate language for their role level
3. Includes relevant medical context
4. Cites information sources when available
5. Provides actionable insights when appropriate

Answer:
""")
    
    async def rewrite_query(self, query: str, user_role: str) -> str:
        """Rewrite query based on user role for better retrieval"""
        try:
            response = await asyncio.to_thread(
                self.llm.invoke,
                self.query_rewriter_prompt.format(role=user_role, query=query)
            )
            rewritten = response.content.strip()
            
            # Fallback to original if rewriting fails
            if len(rewritten) < 3 or "rewritten query" in rewritten.lower():
                return query
                
            logger.info(f"Query rewritten: '{query}' -> '{rewritten}'")
            return rewritten
            
        except Exception as e:
            logger.error(f"Query rewriting failed: {e}")
            return query
    
    async def get_cached_documents(self, user_role: str) -> List[Dict]:
        """Get cached documents for BM25, refresh if needed"""
        current_time = time.time()
        cache_key = f"{user_role}_docs"
        
        # Check if cache is valid
        if (cache_key in self.bm25_docs_cache and 
            current_time - self.last_cache_update < self.cache_ttl):
            return self.bm25_docs_cache[cache_key]
        
        # Refresh cache
        try:
            # Use a dummy vector to get all docs for role
            dummy_vector = [0.0] * 768
            results = await asyncio.to_thread(
                self.index.query,
                vector=dummy_vector,
                top_k=10000,  # Large number to get many docs
                include_metadata=True,
                filter={"role": user_role}
            )
            
            docs = []
            for match in results.get("matches", []):
                metadata = match.get("metadata", {})
                if metadata.get("text"):
                    docs.append({
                        "id": match["id"],
                        "text": metadata["text"],
                        "source": metadata.get("source", ""),
                        "metadata": metadata
                    })
            
            # Update cache
            self.bm25_docs_cache[cache_key] = docs
            self.last_cache_update = current_time
            
            logger.info(f"Document cache updated for role {user_role}: {len(docs)} documents")
            return docs
            
        except Exception as e:
            logger.error(f"Failed to get cached documents: {e}")
            return self.bm25_docs_cache.get(cache_key, [])
    
    async def hybrid_retrieval(self, query: str, user_role: str, top_k: int = 5) -> List[Dict]:
        """Hybrid retrieval combining semantic search and BM25"""
        
        # 1. Semantic Search (Pinecone)
        semantic_results = []
        try:
            embedding = await asyncio.to_thread(self.embed_model.embed_query, query)
            pinecone_results = await asyncio.to_thread(
                self.index.query,
                vector=embedding,
                top_k=top_k * 2,  # Get more for better selection
                include_metadata=True,
                filter={"role": user_role}
            )
            
            for match in pinecone_results.get("matches", []):
                semantic_results.append({
                    "id": match["id"],
                    "text": match["metadata"].get("text", ""),
                    "source": match["metadata"].get("source", ""),
                    "semantic_score": float(match["score"]),
                    "bm25_score": 0.0,
                    "metadata": match["metadata"]
                })
                
        except Exception as e:
            logger.error(f"Semantic search failed: {e}")
        
        # 2. BM25 Search
        bm25_results = []
        try:
            docs = await self.get_cached_documents(user_role)
            if docs and len(docs) > 0:
                texts = [doc["text"] for doc in docs]
                tokenized_texts = [text.lower().split() for text in texts]
                
                if tokenized_texts:
                    bm25 = BM25Okapi(tokenized_texts)
                    query_tokens = query.lower().split()
                    scores = bm25.get_scores(query_tokens)
                    
                    # Get top scoring documents
                    top_indices = np.argsort(scores)[-top_k:][::-1]
                    
                    for idx in top_indices:
                        if scores[idx] > 0:  # Only include relevant results
                            doc = docs[idx]
                            bm25_results.append({
                                "id": doc["id"],
                                "text": doc["text"],
                                "source": doc["source"],
                                "semantic_score": 0.0,
                                "bm25_score": float(scores[idx]),
                                "metadata": doc["metadata"]
                            })
                            
        except Exception as e:
            logger.error(f"BM25 search failed: {e}")
        
        # 3. Combine and rank results
        return self.combine_and_rank_results(semantic_results, bm25_results, top_k)
    
    def combine_and_rank_results(self, semantic_results: List[Dict], 
                                 bm25_results: List[Dict], top_k: int) -> List[Dict]:
        """Combine semantic and BM25 results with hybrid scoring"""
        
        # Combine results by document ID
        combined_docs = {}
        
        # Add semantic results
        for doc in semantic_results:
            doc_id = doc["id"]
            combined_docs[doc_id] = doc
        
        # Add/merge BM25 results
        for doc in bm25_results:
            doc_id = doc["id"]
            if doc_id in combined_docs:
                combined_docs[doc_id]["bm25_score"] = doc["bm25_score"]
            else:
                combined_docs[doc_id] = doc
        
        # Calculate hybrid scores (weighted combination)
        for doc in combined_docs.values():
            # Normalize scores (simple min-max normalization)
            semantic_weight = 0.7
            bm25_weight = 0.3
            
            hybrid_score = (semantic_weight * doc["semantic_score"] + 
                           bm25_weight * doc["bm25_score"])
            doc["hybrid_score"] = hybrid_score
        
        # Sort by hybrid score and return top_k
        sorted_docs = sorted(
            combined_docs.values(),
            key=lambda x: x["hybrid_score"],
            reverse=True
        )
        
        return sorted_docs[:top_k]
    
    async def extract_entities(self, text: str) -> List[str]:
        """Extract medical entities from text"""
        try:
            response = await asyncio.to_thread(
                self.llm.invoke,
                self.entity_prompt.format(text=text)
            )
            
            entities_text = response.content.strip()
            entities = [e.strip() for e in entities_text.split(',') if e.strip()]
            
            # Clean and validate entities
            valid_entities = []
            for entity in entities:
                if len(entity) > 2 and len(entity) < 50:  # Reasonable length
                    valid_entities.append(entity.lower())
            
            return valid_entities[:10]  # Limit number of entities
            
        except Exception as e:
            logger.error(f"Entity extraction failed: {e}")
            return []
    
    async def update_knowledge_graph(self, documents: List[Dict]):
        """Update knowledge graph with new document relationships"""
        try:
            for doc in documents[-5:]:  # Process only recent docs for performance
                entities = await self.extract_entities(doc["text"])
                
                # Add entities as nodes
                for entity in entities:
                    if not self.knowledge_graph.has_node(entity):
                        self.knowledge_graph.add_node(
                            entity,
                            type="medical_entity",
                            first_seen=datetime.utcnow(),
                            frequency=1
                        )
                    else:
                        # Update frequency
                        self.knowledge_graph.nodes[entity]["frequency"] += 1
                
                # Add co-occurrence relationships
                for i, entity1 in enumerate(entities):
                    for entity2 in entities[i+1:]:
                        edge_weight = 1.0
                        
                        if self.knowledge_graph.has_edge(entity1, entity2):
                            # Strengthen existing relationship
                            current_weight = self.knowledge_graph[entity1][entity2]["weight"]
                            edge_weight = min(current_weight + 0.1, 1.0)
                        
                        self.knowledge_graph.add_edge(
                            entity1, entity2,
                            weight=edge_weight,
                            type="co_occurrence",
                            source=doc["source"],
                            last_updated=datetime.utcnow()
                        )
                        
        except Exception as e:
            logger.error(f"Knowledge graph update failed: {e}")
    
    async def graph_enhanced_retrieval(self, entities: List[str]) -> Dict:
        """Get related information using knowledge graph"""
        graph_context = []
        
        try:
            for entity in entities[:5]:  # Limit for performance
                if entity in self.knowledge_graph:
                    # Get direct neighbors
                    neighbors = list(self.knowledge_graph.neighbors(entity))[:3]
                    
                    for neighbor in neighbors:
                        edge_data = self.knowledge_graph[entity][neighbor]
                        graph_context.append({
                            "relationship": f"{entity} ↔ {neighbor}",
                            "type": edge_data.get("type", "related"),
                            "strength": edge_data.get("weight", 0.5),
                            "source": edge_data.get("source", "knowledge_graph")
                        })
            
            return {
                "entities_found": entities,
                "relationships": graph_context,
                "graph_stats": {
                    "total_nodes": len(self.knowledge_graph.nodes()),
                    "total_edges": len(self.knowledge_graph.edges())
                }
            }
            
        except Exception as e:
            logger.error(f"Graph enhanced retrieval failed: {e}")
            return {"entities_found": entities, "relationships": [], "graph_stats": {}}
    
    async def multi_hop_retrieval(self, query: str, user_role: str, max_hops: int = 3) -> Dict:
        """Multi-hop retrieval with chain-of-thought reasoning"""
        
        steps = []
        all_contexts = []
        current_query = query
        
        for hop in range(max_hops):
            # Retrieve documents for current query
            results = await self.hybrid_retrieval(current_query, user_role, top_k=3)
            
            if not results:
                break
            
            step_context = "\n".join([r["text"][:500] for r in results])  # Limit context size
            all_contexts.extend(results)
            
            # Analyze step and determine next action
            step_analysis = await asyncio.to_thread(
                self.llm.invoke,
                self.multi_hop_prompt.format(
                    query=query,
                    step=hop + 1,
                    step_description=f"Searching for: {current_query}",
                    context=step_context
                )
            )
            
            analysis_content = step_analysis.content
            
            # Parse analysis
            needs_more = "YES" in analysis_content and "NEEDS_MORE:" in analysis_content
            
            steps.append({
                "hop": hop + 1,
                "query": current_query,
                "results_count": len(results),
                "analysis": analysis_content,
                "needs_more": needs_more
            })
            
            # Stop if no more information is needed
            if not needs_more:
                break
            
            # Extract next query
            if "NEXT_QUERY:" in analysis_content:
                lines = analysis_content.split('\n')
                next_query_lines = [line for line in lines if "NEXT_QUERY:" in line]
                if next_query_lines:
                    next_query = next_query_lines[0].split("NEXT_QUERY:")[-1].strip()
                    if len(next_query) > 3:
                        current_query = next_query
                    else:
                        break
                else:
                    break
            else:
                break
        
        return {
            "steps": steps,
            "all_contexts": all_contexts,
            "total_hops": len(steps)
        }
    
    async def generate_final_answer(self, query: str, user_role: str, 
                                   contexts: List[Dict], graph_data: Dict, 
                                   multihop_data: Dict) -> str:
        """Generate comprehensive final answer"""
        
        try:
            # Prepare context
            context_text = "\n\n".join([
                f"Source: {ctx['source']}\nContent: {ctx['text'][:400]}"
                for ctx in contexts[:5]
            ])
            
            # Prepare graph context
            graph_context = "\n".join([
                f"• {rel['relationship']} ({rel['type']})"
                for rel in graph_data.get("relationships", [])[:5]
            ])
            
            # Prepare multihop summary
            multihop_summary = "\n".join([
                f"Step {step['hop']}: {step['analysis'][:200]}..."
                for step in multihop_data.get("steps", [])
            ])
            
            # Generate final answer
            response = await asyncio.to_thread(
                self.llm.invoke,
                self.final_answer_prompt.format(
                    query=query,
                    role=user_role,
                    context=context_text,
                    graph_context=graph_context,
                    multihop_summary=multihop_summary
                )
            )
            
            return response.content
            
        except Exception as e:
            logger.error(f"Final answer generation failed: {e}")
            return "I apologize, but I encountered an error while generating a comprehensive answer. Please try rephrasing your question."
    
    async def advanced_query(self, query: str, user_role: str) -> Dict:
        """Main advanced RAG query method orchestrating all techniques"""
        
        start_time = time.time()
        
        try:
            # Step 1: Query Rewriting
            rewritten_query = await self.rewrite_query(query, user_role)
            
            # Step 2: Multi-hop Retrieval
            multihop_results = await self.multi_hop_retrieval(rewritten_query, user_role)
            
            # Step 3: Extract entities for graph retrieval
            entities = await self.extract_entities(query)
            
            # Step 4: Graph-enhanced retrieval
            graph_results = await self.graph_enhanced_retrieval(entities)
            
            # Step 5: Update knowledge graph with new contexts
            await self.update_knowledge_graph(multihop_results["all_contexts"])
            
            # Step 6: Generate final comprehensive answer
            final_answer = await self.generate_final_answer(
                query, user_role,
                multihop_results["all_contexts"],
                graph_results,
                multihop_results
            )
            
            # Collect sources
            sources = list(set([
                ctx["source"] for ctx in multihop_results["all_contexts"]
                if ctx.get("source")
            ]))
            
            processing_time = time.time() - start_time
            
            return {
                "original_query": query,
                "rewritten_query": rewritten_query,
                "answer": final_answer,
                "sources": sources,
                "multi_hop_steps": multihop_results["total_hops"],
                "graph_entities": graph_results["entities_found"],
                "graph_relationships": len(graph_results["relationships"]),
                "processing_time": round(processing_time, 2),
                "retrieval_method": "Advanced Multi-vector Hybrid + Graph + Multi-hop",
                "total_documents_found": len(multihop_results["all_contexts"]),
                "knowledge_graph_stats": graph_results.get("graph_stats", {})
            }
            
        except Exception as e:
            logger.error(f"Advanced RAG query failed: {e}")
            return {
                "original_query": query,
                "rewritten_query": query,
                "answer": f"I encountered an error while processing your advanced query: {str(e)}. Please try again or contact support if the issue persists.",
                "sources": [],
                "multi_hop_steps": 0,
                "graph_entities": [],
                "graph_relationships": 0,
                "processing_time": 0,
                "retrieval_method": "Advanced Multi-vector Hybrid + Graph + Multi-hop",
                "error": str(e)
            }

# Global instance
rag_engine = AdvancedRAGEngine()
