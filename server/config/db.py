import os
from dotenv import load_dotenv
from pymongo import MongoClient
from motor.motor_asyncio import AsyncIOMotorClient
from typing import Optional
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

# MongoDB configuration
MONGO_URL = os.getenv("MONGO_URL", "mongodb://localhost:27017")
DB_NAME = os.getenv("DB_NAME", "healthcare_rag_db")

# Synchronous MongoDB client (for FastAPI routes)
try:
    sync_client = MongoClient(MONGO_URL)
    sync_db = sync_client[DB_NAME]
    
    # Test connection
    sync_client.admin.command('ping')
    logger.info("✅ Connected to MongoDB (sync)")
    
except Exception as e:
    logger.error(f"❌ Failed to connect to MongoDB (sync): {e}")
    sync_client = None
    sync_db = None

# Collections - Fix the boolean evaluation issue
usercollection = sync_db["users"] if sync_db is not None else None
documentscollection = sync_db["documents"] if sync_db is not None else None
graphcollection = sync_db["knowledge_graph"] if sync_db is not None else None
query_analytics_collection = sync_db["query_analytics"] if sync_db is not None else None

# Asynchronous MongoDB client (for async operations)
try:
    async_client = AsyncIOMotorClient(MONGO_URL)
    async_db = async_client[DB_NAME]
    logger.info("✅ Connected to MongoDB (async)")
    
except Exception as e:
    logger.error(f"❌ Failed to connect to MongoDB (async): {e}")
    async_client = None
    async_db = None

class DatabaseManager:
    """Enhanced database manager for advanced operations"""
    
    def __init__(self):
        self.sync_db = sync_db
        self.async_db = async_db
    
    async def store_document_metadata(self, doc_metadata: dict) -> str:
        """Store document metadata"""
        try:
            if self.async_db is None:
                raise Exception("Database not connected")
                
            result = await self.async_db.documents.insert_one(doc_metadata)
            return str(result.inserted_id)
        except Exception as e:
            logger.error(f"Failed to store document metadata: {e}")
            raise
    
    async def get_user_documents(self, user_role: str, user_id: Optional[str] = None):
        """Get documents accessible to user role"""
        try:
            if self.async_db is None:
                return []
                
            query = {"accessible_roles": user_role}
            if user_role == "patient" and user_id:
                query["patient_id"] = user_id
                
            cursor = self.async_db.documents.find(query)
            return await cursor.to_list(length=None)
        except Exception as e:
            logger.error(f"Failed to get user documents: {e}")
            return []
    
    async def store_query_analytics(self, analytics_data: dict):
        """Store query analytics for insights"""
        try:
            if self.async_db is None:
                return
                
            analytics_data["timestamp"] = analytics_data.get("timestamp", datetime.utcnow())
            await self.async_db.query_analytics.insert_one(analytics_data)
        except Exception as e:
            logger.error(f"Failed to store query analytics: {e}")
    
    async def get_query_analytics(self, user_role: str, username: str, limit: int = 100):
        """Get query analytics for user"""
        try:
            if self.async_db is None:
                return []
                
            cursor = self.async_db.query_analytics.find(
                {"username": username, "user_role": user_role}
            ).sort("timestamp", -1).limit(limit)
            
            return await cursor.to_list(length=limit)
        except Exception as e:
            logger.error(f"Failed to get query analytics: {e}")
            return []
    
    async def store_graph_data(self, graph_data: dict):
        """Store knowledge graph data"""
        try:
            if self.async_db is None:
                return
                
            await self.async_db.knowledge_graph.insert_one(graph_data)
        except Exception as e:
            logger.error(f"Failed to store graph data: {e}")
    
    def get_collection_stats(self):
        """Get database collection statistics"""
        try:
            if self.sync_db is None:
                return {}
                
            stats = {}
            collections = ["users", "documents", "knowledge_graph", "query_analytics"]
            
            for collection_name in collections:
                collection = self.sync_db[collection_name]
                stats[collection_name] = collection.count_documents({})
                
            return stats
        except Exception as e:
            logger.error(f"Failed to get collection stats: {e}")
            return {}

# Global database manager instance
db_manager = DatabaseManager()

# Database initialization
def init_database():
    """Initialize database with indexes and default data"""
    try:
        if sync_db is None:
            logger.error("Database not connected, skipping initialization")
            return
            
        # Create indexes for better performance
        if usercollection is not None:
            usercollection.create_index("username", unique=True)
        if documentscollection is not None:
            documentscollection.create_index("doc_id")
            documentscollection.create_index("role")
            documentscollection.create_index("accessible_roles")
        if query_analytics_collection is not None:
            query_analytics_collection.create_index("username")
            query_analytics_collection.create_index("timestamp")
        
        logger.info("✅ Database indexes created successfully")
        
        # Create default admin user if not exists
        if usercollection is not None:
            admin_exists = usercollection.find_one({"username": "admin"})
            if not admin_exists:
                from auth.hash_utils import hash_password
                
                default_admin = {
                    "username": "admin",
                    "password": hash_password("admin123"),
                    "role": "admin",
                    "full_name": "System Administrator",
                    "created_at": datetime.utcnow(),
                    "is_active": True
                }
                
                usercollection.insert_one(default_admin)
                logger.info("✅ Default admin user created (username: admin, password: admin123)")
            
    except Exception as e:
        logger.error(f"Failed to initialize database: {e}")

# Initialize database on import
init_database()
