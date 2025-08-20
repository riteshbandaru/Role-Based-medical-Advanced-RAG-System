from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.responses import JSONResponse
from fastapi.encoders import jsonable_encoder
from typing import Dict, List
from datetime import datetime, timedelta

from .models import User, UserProfile, UserRole, LoginRequest, TokenResponse
from .hash_utils import hash_password, verify_password, create_access_token, verify_token
from config.db import usercollection

router = APIRouter()
security = HTTPBearer(auto_error=False)

def get_user_permissions(role: str) -> List[str]:
    """Get permissions based on user role"""
    permission_map = {
        "admin": [
            "upload_docs", "manage_users", "view_all", "delete_docs",
            "graph_admin", "system_analytics", "user_management", "graph_query"
        ],
        "doctor": [
            "view_patient_data", "view_medical_reports", "upload_reports",
            "graph_query", "patient_management", "prescription_access"
        ],
        "nurse": [
            "view_patient_basic", "view_medications", "update_patient_status",
            "graph_query", "care_protocols", "shift_reports"
        ],
        "patient": [
            "view_own_data", "basic_query", "appointment_info",
            "personal_health", "medication_reminders"
        ]
    }
    return permission_map.get(role, [])

async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)) -> UserProfile:
    """Get current authenticated user"""
    if not credentials:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication credentials required",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    token = credentials.credentials
    payload = verify_token(token)
    
    if not payload:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    username = payload.get("username")
    if not username:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token payload"
        )
    
    # Get user from database with proper None check
    if usercollection is None:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Database connection unavailable"
        )
    
    user = usercollection.find_one({"username": username})
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found"
        )
    
    return UserProfile(
        username=user["username"],
        role=user["role"],
        specialization=user.get("specialization"),
        department=user.get("department"),
        patient_id=user.get("patient_id"),
        full_name=user.get("full_name"),
        permissions=get_user_permissions(user["role"]),
        created_at=user.get("created_at")
    )

@router.post("/signup")
async def signup(user_data: User):
    """Register a new user"""
    # Check database connection
    if usercollection is None:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Database connection unavailable"
        )
    
    # Check if username already exists
    existing_user = usercollection.find_one({"username": user_data.username})
    if existing_user:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Username already registered"
        )
    
    # Create user document
    user_doc = {
        "username": user_data.username,
        "password": hash_password(user_data.password),
        "role": user_data.role,
        "specialization": user_data.specialization,
        "department": user_data.department,
        "patient_id": user_data.patient_id,
        "full_name": user_data.full_name,
        "created_at": datetime.utcnow(),
        "updated_at": datetime.utcnow(),
        "is_active": True
    }
    
    # Insert user into database
    try:
        result = usercollection.insert_one(user_doc)
        if result.inserted_id:
            response_data = {
                "message": "User created successfully",
                "username": user_data.username,
                "role": user_data.role
            }
            return JSONResponse(content=jsonable_encoder(response_data))
        else:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to create user"
            )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Database error: {str(e)}"
        )

@router.post("/login")
async def login(login_data: LoginRequest):
    """Authenticate user and return access token - FIXED JSON SERIALIZATION"""
    # Check database connection
    if usercollection is None:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Database connection unavailable"
        )
    
    # Find user in database
    user = usercollection.find_one({"username": login_data.username})
    
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid username or password"
        )
    
    # Verify password
    if not verify_password(login_data.password, user["password"]):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid username or password"
        )
    
    # Check if user is active
    if not user.get("is_active", True):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Account is deactivated"
        )
    
    # Create access token
    token_data = {
        "username": user["username"],
        "role": user["role"],
        "user_id": str(user["_id"])
    }
    
    access_token = create_access_token(
        data=token_data,
        expires_delta=timedelta(hours=24)
    )
    
    # Create user profile - Convert datetime to string
    user_profile_data = {
        "username": user["username"],
        "role": user["role"],
        "specialization": user.get("specialization"),
        "department": user.get("department"),
        "patient_id": user.get("patient_id"),
        "full_name": user.get("full_name"),
        "permissions": get_user_permissions(user["role"]),
        "created_at": user.get("created_at").isoformat() if user.get("created_at") else None
    }
    
    # Update last login
    usercollection.update_one(
        {"_id": user["_id"]},
        {"$set": {"last_login": datetime.utcnow()}}
    )
    
    # Prepare response data
    response_data = {
        "access_token": access_token,
        "token_type": "bearer",
        "expires_in": 86400,
        "user": user_profile_data
    }
    
    # Use jsonable_encoder to handle any remaining datetime objects
    return JSONResponse(content=jsonable_encoder(response_data))

@router.get("/me")
async def get_current_user_info(current_user: UserProfile = Depends(get_current_user)):
    """Get current user information"""
    return JSONResponse(content=jsonable_encoder(current_user.dict()))

@router.get("/permissions")
async def get_user_permissions_endpoint(current_user: UserProfile = Depends(get_current_user)):
    """Get current user permissions"""
    response_data = {
        "username": current_user.username,
        "role": current_user.role,
        "permissions": current_user.permissions
    }
    return JSONResponse(content=jsonable_encoder(response_data))

@router.post("/logout")
async def logout(current_user: UserProfile = Depends(get_current_user)):
    """Logout user (client-side token removal)"""
    response_data = {
        "message": f"User {current_user.username} logged out successfully",
        "timestamp": datetime.utcnow().isoformat()
    }
    return JSONResponse(content=jsonable_encoder(response_data))
