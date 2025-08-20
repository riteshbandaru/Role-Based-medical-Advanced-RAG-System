from pydantic import BaseModel, Field
from typing import Optional, List
from enum import Enum
from datetime import datetime

class UserRole(str, Enum):
    ADMIN = "admin"
    DOCTOR = "doctor"
    NURSE = "nurse"
    PATIENT = "patient"

class User(BaseModel):
    username: str = Field(..., min_length=3, max_length=50)
    password: str = Field(..., min_length=6)
    role: UserRole
    specialization: Optional[str] = Field(None, max_length=100)
    department: Optional[str] = Field(None, max_length=100)
    patient_id: Optional[str] = Field(None, max_length=50)
    full_name: Optional[str] = Field(None, max_length=100)

class UserProfile(BaseModel):
    username: str
    role: UserRole
    specialization: Optional[str] = None
    department: Optional[str] = None
    patient_id: Optional[str] = None
    full_name: Optional[str] = None
    permissions: List[str] = []
    created_at: Optional[datetime] = None

class LoginRequest(BaseModel):
    username: str
    password: str

class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    expires_in: int = 86400
    user: UserProfile
