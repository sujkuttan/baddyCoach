import os
from typing import Optional
from fastapi import HTTPException, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials


class Security:
    def __init__(self):
        self.bearer_scheme = HTTPBearer()
        self.api_keys = self._load_api_keys()

    def _load_api_keys(self) -> dict:
        """Load API keys from environment variables."""
        api_keys = {}
        for key, value in os.environ.items():
            if key.startswith("API_KEY_"):
                user_id = key.split("_")[1]
                api_keys[user_id] = value
        return api_keys

    async def authenticate_request(self, request: Request) -> str:
        """Authenticate request using API key."""
        # Check for API key in header
        api_key = request.headers.get("X-API-Key")
        if not api_key:
            raise HTTPException(status_code=401, detail="Missing API key")

        # Validate API key
        user_id = self._validate_api_key(api_key)
        if not user_id:
            raise HTTPException(status_code=401, detail="Invalid API key")

        return user_id

    def _validate_api_key(self, api_key: str) -> Optional[str]:
        """Validate API key and return user ID."""
        for user_id, key in self.api_keys.items():
            if key == api_key:
                return user_id
        return None

    async def authenticate_request_with_bearer(self, request: Request, credentials: HTTPAuthorizationCredentials = None):
        """Authenticate request using Bearer token."""
        if not credentials:
            raise HTTPException(status_code=401, detail="Missing Bearer token")

        user_id = self._validate_api_key(credentials.credentials)
        if not user_id:
            raise HTTPException(status_code=401, detail="Invalid API key")

        return user_id

    def require_auth(self, func):
        """Decorator to require authentication."""
        async def wrapper(*args, **kwargs):
            request = args[0] if args else None
            if request:
                await self.authenticate_request(request)
            return await func(*args, **kwargs)
        return wrapper

    def require_bearer_auth(self, func):
        """Decorator to require Bearer token authentication."""
        async def wrapper(*args, **kwargs):
            request = args[0] if args else None
            if request:
                await self.authenticate_request_with_bearer(request)
            return await func(*args, **kwargs)
        return wrapper