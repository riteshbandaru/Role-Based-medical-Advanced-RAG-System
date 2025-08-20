from fastapi import APIRouter, Depends, UploadFile, File, Form, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from fastapi.encoders import jsonable_encoder
from typing import List, Optional
import uuid
import logging
from datetime import datetime
import json

from auth.routes import get_current_user
from auth.models import UserProfile
from .vector_database import advanced_vectorstore

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

router = APIRouter()

def serialize_pinecone_stats(stats_obj):
    """Convert Pinecone stats to JSON-serializable format"""
    try:
        if hasattr(stats_obj, '__dict__'):
            # Convert object to dict
            stats_dict = {}
            for key, value in stats_obj.__dict__.items():
                try:
                    json.dumps(value)  # Test if serializable
                    stats_dict[key] = value
                except (TypeError, ValueError):
                    if hasattr(value, '__dict__'):
                        stats_dict[key] = str(value)
                    else:
                        stats_dict[key] = str(value)
            return stats_dict
        else:
            return str(stats_obj)
    except Exception as e:
        logger.error(f"Failed to serialize stats: {e}")
        return {"error": str(e)}

@router.post("/upload")
async def upload_documents(
    background_tasks: BackgroundTasks,
    user: UserProfile = Depends(get_current_user),
    files: List[UploadFile] = File(...),
    role: str = Form(...),
    doc_type: str = Form(default="medical_report"),
    priority: str = Form(default="medium")
):
    """Upload and process documents with advanced techniques"""
    
    try:
        logger.info(f"Document upload request from user {user.username}")
        
        # Check permissions
        if "upload_docs" not in user.permissions:
            raise HTTPException(
                status_code=403,
                detail="Insufficient permissions to upload documents"
            )
        
        # Validate inputs
        if not files:
            raise HTTPException(status_code=400, detail="No files provided")
        
        valid_roles = ["admin", "doctor", "nurse", "patient"]
        if role not in valid_roles:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid role. Must be one of: {valid_roles}"
            )
        
        valid_types = ["medical_report", "patient_record", "treatment_protocol", 
                       "medication_guide", "research_paper", "care_plan", "other"]
        if doc_type not in valid_types:
            doc_type = "other"
        
        valid_priorities = ["high", "medium", "low"]
        if priority not in valid_priorities:
            priority = "medium"
        
        # Validate file types and sizes
        for file in files:
            if not file.filename or not file.filename.lower().endswith('.pdf'):
                raise HTTPException(
                    status_code=400,
                    detail=f"Unsupported file type: {file.filename}. Only PDF files are allowed."
                )
            
            content = await file.read()
            await file.seek(0)
            
            if len(content) > 10 * 1024 * 1024:
                raise HTTPException(
                    status_code=400,
                    detail=f"File {file.filename} is too large. Maximum size is 10MB."
                )
        
        # Generate unique document ID
        doc_id = str(uuid.uuid4())
        
        logger.info(f"Processing {len(files)} documents for user {user.username}, doc_id: {doc_id}")
        
        # Process documents
        processing_stats = await advanced_vectorstore.process_and_store_documents(
            files, role, doc_id, doc_type, priority
        )
        
        # Prepare response
        uploaded_files = [{"filename": f.filename, "size": len(await f.read())} for f in files]
        for f in files:
            await f.seek(0)
        
        response_data = {
            "message": "Documents uploaded and processed successfully",
            "doc_id": doc_id,
            "uploaded_by": user.username,
            "accessible_to_role": role,
            "document_type": doc_type,
            "priority": priority,
            "upload_timestamp": datetime.utcnow().isoformat(),
            "files": uploaded_files,
            "processing_stats": processing_stats
        }
        
        logger.info(f"✅ Document upload completed successfully for doc_id: {doc_id}")
        return JSONResponse(content=jsonable_encoder(response_data))
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Document upload failed for user {user.username}: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={
                "error": f"Document processing failed: {str(e)}",
                "message": "Please try again with smaller files or contact support.",
                "doc_id": None
            }
        )

@router.post("/search")
async def search_documents(
    user: UserProfile = Depends(get_current_user),
    query: str = Form(...),
    doc_type: Optional[str] = Form(None),
    limit: int = Form(default=5)
):
    """Search documents using semantic similarity"""
    
    try:
        logger.info(f"Document search request from user {user.username}: {query[:50]}...")
        
        if not query or not query.strip():
            raise HTTPException(status_code=400, detail="Query cannot be empty")
        
        if limit < 1 or limit > 20:
            limit = 5
        
        # Search for similar documents
        results = await advanced_vectorstore.search_similar_documents(
            query.strip(), user.role, doc_type, limit
        )
        
        response_data = {
            "query": query,
            "user_role": user.role,
            "doc_type_filter": doc_type,
            "results_count": len(results),
            "results": results,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        logger.info(f"Document search successful for user {user.username}: {len(results)} results")
        return JSONResponse(content=jsonable_encoder(response_data))
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Document search failed for user {user.username}: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={
                "error": f"Document search failed: {str(e)}",
                "query": query,
                "results": [],
                "results_count": 0
            }
        )

@router.get("/stats")
async def get_document_stats(user: UserProfile = Depends(get_current_user)):
    """Get document and index statistics - FIXED JSON SERIALIZATION"""
    
    try:
        # Get vector store statistics with proper serialization
        raw_index_stats = advanced_vectorstore.get_index_stats()
        
        # Safely serialize index stats
        index_stats = {}
        for key, value in raw_index_stats.items():
            try:
                # Test if value is JSON serializable
                json.dumps(value)
                index_stats[key] = value
            except (TypeError, ValueError):
                # Convert non-serializable objects to strings
                if hasattr(value, '__dict__'):
                    index_stats[key] = serialize_pinecone_stats(value)
                else:
                    index_stats[key] = str(value)
        
        # Get database statistics
        from config.db import db_manager
        db_stats = db_manager.get_collection_stats()
        
        response_data = {
            "user": user.username,
            "role": user.role,
            "timestamp": datetime.utcnow().isoformat(),
            "vector_store": index_stats,
            "database": db_stats,
            "permissions": user.permissions
        }
        
        return JSONResponse(content=jsonable_encoder(response_data))
        
    except Exception as e:
        logger.error(f"Failed to get document stats: {e}")
        return JSONResponse(
            status_code=500,
            content={
                "error": f"Failed to retrieve statistics: {str(e)}",
                "user": user.username if user else "unknown"
            }
        )

@router.get("/types")
async def get_document_types():
    """Get available document types"""
    
    document_types = {
        "medical_report": {
            "name": "Medical Report",
            "description": "Clinical reports, test results, diagnostic reports",
            "typical_users": ["doctor", "nurse"]
        },
        "patient_record": {
            "name": "Patient Record",
            "description": "Patient history, demographics, medical background",
            "typical_users": ["doctor", "nurse"]
        },
        "treatment_protocol": {
            "name": "Treatment Protocol",
            "description": "Standard care procedures, treatment guidelines",
            "typical_users": ["doctor", "nurse"]
        },
        "medication_guide": {
            "name": "Medication Guide",
            "description": "Drug information, dosage instructions, side effects",
            "typical_users": ["doctor", "nurse", "patient"]
        },
        "research_paper": {
            "name": "Research Paper",
            "description": "Medical research, clinical studies, scientific papers",
            "typical_users": ["doctor", "admin"]
        },
        "care_plan": {
            "name": "Care Plan",
            "description": "Patient care plans, nursing protocols, recovery plans",
            "typical_users": ["nurse", "doctor"]
        },
        "other": {
            "name": "Other",
            "description": "Other medical documents not fitting above categories",
            "typical_users": ["admin", "doctor", "nurse"]
        }
    }
    
    return JSONResponse(content={
        "document_types": document_types,
        "roles": ["admin", "doctor", "nurse", "patient"],
        "priorities": ["high", "medium", "low"]
    })

@router.get("/health")
async def docs_health():
    """Health check for document services"""
    
    try:
        # Test vector store connection with safe serialization
        index_stats = advanced_vectorstore.get_index_stats()
        
        # Convert to safe format
        safe_stats = {}
        for key, value in index_stats.items():
            try:
                json.dumps(value)
                safe_stats[key] = value
            except (TypeError, ValueError):
                safe_stats[key] = str(value)
        
        return JSONResponse(content={
            "status": "healthy",
            "timestamp": datetime.utcnow().isoformat(),
            "components": {
                "vector_store": "operational",
                "document_processor": "operational",
                "embedding_model": "operational"
            },
            "stats": safe_stats
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
