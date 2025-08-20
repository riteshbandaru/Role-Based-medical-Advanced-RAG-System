from fastapi import APIRouter, Depends, Form, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from fastapi.encoders import jsonable_encoder
from auth.routes import get_current_user
from auth.models import UserProfile
from .advanced_rag import rag_engine
from .chat_query import simple_rag
from config.db import db_manager
from datetime import datetime
from typing import Dict, Any
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

router = APIRouter()

async def store_query_analytics_background(analytics_data: Dict[str, Any]):
    """Store query analytics in background"""
    try:
        await db_manager.store_query_analytics(analytics_data)
    except Exception as e:
        logger.error(f"Failed to store analytics: {e}")

@router.post("/advanced")
async def advanced_chat(
    background_tasks: BackgroundTasks,
    message: str = Form(...),
    user: UserProfile = Depends(get_current_user)
):
    """Advanced RAG chat with all techniques"""
    
    try:
        logger.info(f"Advanced chat request from user {user.username}: {message[:50]}...")
        
        if not message or not message.strip():
            raise HTTPException(status_code=400, detail="Message cannot be empty")
        
        # Execute advanced RAG query
        result = await rag_engine.advanced_query(message.strip(), user.role)
        
        # Prepare response
        response_data = {
            "user": user.username,
            "role": user.role,
            "timestamp": datetime.utcnow().isoformat(),
            **result
        }
        
        # Store analytics in background
        analytics_data = {
            "username": user.username,
            "user_role": user.role,
            "query": message,
            "query_type": "advanced",
            "processing_time": result.get("processing_time", 0),
            "sources_found": len(result.get("sources", [])),
            "multihop_steps": result.get("multi_hop_steps", 0),
            "timestamp": datetime.utcnow(),
            "success": "error" not in result
        }
        
        background_tasks.add_task(store_query_analytics_background, analytics_data)
        
        logger.info(f"Advanced chat successful for user {user.username}")
        return JSONResponse(content=jsonable_encoder(response_data))
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Advanced chat error for user {user.username}: {str(e)}")
        return JSONResponse(
            status_code=500,
            content=jsonable_encoder({
                "user": user.username,
                "role": user.role,
                "timestamp": datetime.utcnow().isoformat(),
                "original_query": message,
                "answer": f"I apologize, but I encountered an error while processing your query: {str(e)}. Please try again.",
                "sources": [],
                "error": str(e),
                "retrieval_method": "Error - Advanced RAG Failed"
            })
        )

@router.post("/simple")
async def simple_chat(
    background_tasks: BackgroundTasks,
    message: str = Form(...),
    user: UserProfile = Depends(get_current_user)
):
    """Simple RAG chat for comparison"""
    
    try:
        logger.info(f"Simple chat request from user {user.username}: {message[:50]}...")
        
        if not message or not message.strip():
            raise HTTPException(status_code=400, detail="Message cannot be empty")
        
        # Execute simple RAG query
        result = await simple_rag.simple_query(message.strip(), user.role)
        
        # Prepare response
        response_data = {
            "user": user.username,
            "role": user.role,
            "timestamp": datetime.utcnow().isoformat(),
            **result
        }
        
        # Store analytics in background
        analytics_data = {
            "username": user.username,
            "user_role": user.role,
            "query": message,
            "query_type": "simple",
            "processing_time": 0,
            "sources_found": result.get("num_sources", 0),
            "multihop_steps": 0,
            "timestamp": datetime.utcnow(),
            "success": "error" not in result
        }
        
        background_tasks.add_task(store_query_analytics_background, analytics_data)
        
        logger.info(f"Simple chat successful for user {user.username}")
        return JSONResponse(content=jsonable_encoder(response_data))
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Simple chat error for user {user.username}: {str(e)}")
        return JSONResponse(
            status_code=500,
            content=jsonable_encoder({
                "user": user.username,
                "role": user.role,
                "timestamp": datetime.utcnow().isoformat(),
                "original_query": message,
                "answer": f"I encountered an error while processing your query: {str(e)}. Please try again.",
                "sources": [],
                "error": str(e),
                "method": "Error - Simple RAG Failed"
            })
        )



@router.get("/capabilities")
async def get_chat_capabilities(user: UserProfile = Depends(get_current_user)):
    """Get available chat capabilities based on user role"""
    
    base_capabilities = [
        "Natural Language Query Processing",
        "Document-based Information Retrieval",
        "Context-aware Responses"
    ]
    
    role_capabilities = {
        "admin": [
            "System-wide Document Access",
            "Advanced Analytics Queries",
            "User Management Queries",
            "System Performance Metrics"
        ],
        "doctor": [
            "Patient Medical Records Access",
            "Clinical Decision Support",
            "Medical Literature Search",
            "Treatment Recommendation Analysis"
        ],
        "nurse": [
            "Patient Care Protocols",
            "Medication Administration Guidance",
            "Shift Report Information",
            "Care Plan Updates"
        ],
        "patient": [
            "Personal Health Information",
            "Appointment Scheduling Help",
            "Basic Medical Information",
            "Health Education Resources"
        ]
    }
    
    advanced_features = [
        "🔄 Hybrid Retrieval (Semantic + Keyword)",
        "🧠 Multi-hop Chain-of-Thought Reasoning",
        "🔄 Query Rewriting & Optimization",
        "🕸️ Knowledge Graph Integration",
        "📊 Real-time Analytics Tracking"
    ]
    
    return JSONResponse(content={
        "user": user.username,
        "role": user.role.title(),
        "base_capabilities": base_capabilities,
        "role_specific": role_capabilities.get(user.role, []),
        "advanced_features": advanced_features,
        "permissions": user.permissions
    })

@router.get("/analytics")
async def get_user_analytics(
    user: UserProfile = Depends(get_current_user),
    limit: int = 50
):
    """Get user's query analytics"""
    
    try:
        analytics = await db_manager.get_query_analytics(
            user.role, user.username, limit
        )
        
        # Calculate summary statistics
        total_queries = len(analytics)
        successful_queries = sum(1 for a in analytics if a.get("success", True))
        avg_processing_time = 0
        
        if analytics:
            processing_times = [a.get("processing_time", 0) for a in analytics]
            avg_processing_time = sum(processing_times) / len(processing_times)
        
        # Query type distribution
        query_types = {}
        for a in analytics:
            query_type = a.get("query_type", "unknown")
            query_types[query_type] = query_types.get(query_type, 0) + 1
        
        return JSONResponse(content={
            "user": user.username,
            "role": user.role,
            "summary": {
                "total_queries": total_queries,
                "successful_queries": successful_queries,
                "success_rate": round(successful_queries / total_queries * 100, 2) if total_queries > 0 else 0,
                "avg_processing_time": round(avg_processing_time, 2)
            },
            "query_types": query_types,
            "recent_queries": analytics[:10]  # Return only recent queries for privacy
        })
        
    except Exception as e:
        logger.error(f"Failed to get user analytics: {e}")
        raise HTTPException(
            status_code=500,
            detail="Failed to retrieve analytics data"
        )

@router.get("/health")
async def chat_health():
    """Health check for chat services"""
    
    try:
        # Test basic components
        components = {
            "advanced_rag": True,
            "simple_rag": True,
            "knowledge_graph": len(rag_engine.knowledge_graph.nodes()) > 0,
            "embedding_model": True,
            "llm": True
        }
        
        return JSONResponse(content={
            "status": "healthy",
            "timestamp": datetime.utcnow().isoformat(),
            "components": components,
            "knowledge_graph_stats": {
                "nodes": len(rag_engine.knowledge_graph.nodes()),
                "edges": len(rag_engine.knowledge_graph.edges())
            }
        })
        
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={
                "status": "unhealthy",
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }
        )
