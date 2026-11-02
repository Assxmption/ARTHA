# Security & Authentication Module

**Status**: Implemented  
**Path**: `app/security/auth.py`, `app/security/__init__.py`  

## Overview
The **Security & Authentication Module** establishes cryptographic access control for ARTHA's API endpoints. It validates JSON Web Tokens (JWT) issued by **Supabase Auth**, extracting user identities, authorization scopes, and session validity.

---

## Architecture & Implementation

### 1. Fail-Closed Token Verification
The FastAPI dependency `get_current_user` guards sensitive API routes (e.g., Paper Trading execution, User Watchlists):
```python
def get_current_user(credentials: HTTPAuthorizationCredentials = Security(security)) -> Dict[str, Any]:
    if not SUPABASE_JWT_SECRET:
        raise HTTPException(
            status_code=500,
            detail="SUPABASE_JWT_SECRET environment variable is not set on the backend."
        )
    token = credentials.credentials
    try:
        payload = jwt.decode(
            token,
            SUPABASE_JWT_SECRET,
            algorithms=["HS256"],
            options={"verify_aud": False}
        )
        return payload
    except JWTError as e:
        raise HTTPException(status_code=401, detail=f"Could not validate credentials: {str(e)}")
```

### Key Principles:
- **Fail-Closed Design**: If the backend environment variable `SUPABASE_JWT_SECRET` is missing or unconfigured, the system immediately returns HTTP 500 rather than bypassing authorization.
- **Algorithm Enforcement**: Explicitly requires HMAC-SHA256 (`HS256`), rejecting unauthenticated `none` algorithm attacks.
- **Decoupled User Identity**: Injected dependencies pass the decoded JWT claims (`sub`, `email`, `role`) directly into route handlers.

---

## Protected Surfaces
1. **Paper Trading Daemon (`app/api/paper_trade_routes.py`)**:
   - `POST /api/paper/start`
   - `POST /api/paper/stop`
   - `POST /api/paper/reset`
   - Prevents unauthorized state alteration of virtual capital or background daemons.
2. **User Watchlists & Portfolio State**:
   - Custom watchlist management stored in Supabase PostgreSQL tables.
3. **CI/CD Keep-Alive Health Checks**:
   - Monitored by GitHub Actions (`.github/workflows/keep-supabase-alive.yml`) to prevent free-tier database sleep.

---

## System Connections
- **Backend API**: Injected across `app/api/paper_trade_routes.py` and `app/api/company_routes.py`.
- **Frontend Auth**: Integrated with Supabase JS SDK via `frontend/src/contexts/AuthContext.jsx` and `frontend/src/pages/Auth.jsx`.
