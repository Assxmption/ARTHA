from fastapi import HTTPException, Security
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from jose import jwt, JWTError
import os
from typing import Dict, Any
from dotenv import load_dotenv

load_dotenv()

# Supabase JWT Secret should be in the environment variables
SUPABASE_JWT_SECRET = os.getenv("SUPABASE_JWT_SECRET")

security = HTTPBearer()

def get_current_user(credentials: HTTPAuthorizationCredentials = Security(security)) -> Dict[str, Any]:
    """
    Validates the JWT token issued by Supabase.
    This dependency can be injected into any FastAPI route to protect it.
    """
    if not SUPABASE_JWT_SECRET:
        # Fail closed if the environment isn't properly configured
        raise HTTPException(
            status_code=500,
            detail="SUPABASE_JWT_SECRET environment variable is not set on the backend."
        )

    token = credentials.credentials
    
    try:
        # Supabase signs JWTs using HS256
        payload = jwt.decode(
            token,
            SUPABASE_JWT_SECRET,
            algorithms=["HS256"],
            options={"verify_aud": False}  # By default, aud is 'authenticated'
        )
        return payload
    except JWTError as e:
        raise HTTPException(
            status_code=401,
            detail=f"Could not validate credentials: {str(e)}",
            headers={"WWW-Authenticate": "Bearer"},
        )
