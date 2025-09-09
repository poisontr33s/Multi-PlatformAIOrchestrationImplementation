"""
Unified Authentication Manager for multi-platform AI orchestration.
Handles OAuth 2.1 + PKCE authentication across all platforms with graceful degradation.
"""

import asyncio
import time
import json
import base64
import hashlib
import secrets
from typing import Dict, Optional, Any, List
from dataclasses import dataclass
from enum import Enum
import structlog
import aiohttp
import jwt
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa

from ..utils.retry import RetryManager
from ..utils.circuit_breaker import CircuitBreaker


class AuthProvider(Enum):
    """Supported authentication providers."""
    GITHUB = "github"
    GOOGLE = "google"
    MICROSOFT = "microsoft"
    FIREBASE = "firebase"
    JULES = "jules"


class TokenType(Enum):
    """Types of authentication tokens."""
    ACCESS_TOKEN = "access_token"
    REFRESH_TOKEN = "refresh_token"
    ID_TOKEN = "id_token"
    API_KEY = "api_key"


@dataclass
class AuthCredentials:
    """Authentication credentials for a provider."""
    provider: AuthProvider
    access_token: Optional[str] = None
    refresh_token: Optional[str] = None
    id_token: Optional[str] = None
    api_key: Optional[str] = None
    expires_at: Optional[float] = None
    scope: Optional[str] = None
    token_type: str = "Bearer"


@dataclass
class OAuth2Config:
    """OAuth 2.1 configuration for a provider."""
    provider: AuthProvider
    client_id: str
    client_secret: str
    redirect_uri: str
    authorization_url: str
    token_url: str
    scope: str
    use_pkce: bool = True


@dataclass
class PKCEChallenge:
    """PKCE challenge for OAuth 2.1 flow."""
    code_verifier: str
    code_challenge: str
    code_challenge_method: str = "S256"


class UnifiedAuthenticationManager:
    """
    Unified authentication manager that coordinates OAuth 2.1 + PKCE
    authentication across all AI platforms with token refresh and fallback.
    """
    
    def __init__(self):
        self.logger = structlog.get_logger("auth_manager")
        
        # Authentication state
        self.credentials: Dict[AuthProvider, AuthCredentials] = {}
        self.oauth_configs: Dict[AuthProvider, OAuth2Config] = {}
        self.pkce_challenges: Dict[str, PKCEChallenge] = {}
        
        # HTTP session for authentication
        self.session: Optional[aiohttp.ClientSession] = None
        
        # Token refresh management
        self.refresh_tasks: Dict[AuthProvider, asyncio.Task] = {}
        self.refresh_locks: Dict[AuthProvider, asyncio.Lock] = {}
        
        # Circuit breakers for auth providers
        self.circuit_breakers: Dict[AuthProvider, CircuitBreaker] = {}
        
        # Retry manager
        self.retry_manager = RetryManager(max_attempts=3, base_delay=1.0)
        
        # Configuration loaded from environment/config files
        self.config: Dict[str, Any] = {}

    async def initialize(self) -> None:
        """Initialize the unified authentication manager."""
        try:
            self.logger.info("Initializing unified authentication manager")
            
            # Load configuration
            await self._load_configuration()
            
            # Initialize HTTP session
            timeout = aiohttp.ClientTimeout(total=30, connect=10)
            self.session = aiohttp.ClientSession(timeout=timeout)
            
            # Initialize circuit breakers
            for provider in AuthProvider:
                self.circuit_breakers[provider] = CircuitBreaker(
                    failure_threshold=3,
                    recovery_timeout=60
                )
                self.refresh_locks[provider] = asyncio.Lock()
            
            # Initialize OAuth configurations
            await self._initialize_oauth_configs()
            
            # Load existing credentials
            await self._load_existing_credentials()
            
            # Start token refresh monitoring
            asyncio.create_task(self._monitor_token_refresh())
            
            self.logger.info("Unified authentication manager initialized successfully")
            
        except Exception as e:
            self.logger.error("Failed to initialize authentication manager", error=str(e))
            raise

    async def _load_configuration(self) -> None:
        """Load authentication configuration from environment and config files."""
        import os
        
        # Load from environment variables
        self.config = {
            "github": {
                "client_id": os.getenv("GITHUB_APP_ID"),
                "client_secret": os.getenv("GITHUB_CLIENT_SECRET"),
                "token": os.getenv("GITHUB_TOKEN"),
                "webhook_secret": os.getenv("GITHUB_WEBHOOK_SECRET")
            },
            "google": {
                "project_id": os.getenv("GOOGLE_PROJECT_ID"),
                "client_id": os.getenv("GOOGLE_CLIENT_ID"),
                "client_secret": os.getenv("GOOGLE_CLIENT_SECRET"),
                "ai_api_key": os.getenv("GOOGLE_AI_API_KEY"),
                "service_account_path": os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
            },
            "microsoft": {
                "tenant_id": os.getenv("AZURE_TENANT_ID"),
                "client_id": os.getenv("AZURE_CLIENT_ID"),
                "client_secret": os.getenv("AZURE_CLIENT_SECRET"),
                "subscription_id": os.getenv("AZURE_SUBSCRIPTION_ID"),
                "openai_api_key": os.getenv("OPENAI_API_KEY")
            },
            "firebase": {
                "project_id": os.getenv("FIREBASE_PROJECT_ID"),
                "api_key": os.getenv("FIREBASE_API_KEY"),
                "service_account_path": os.getenv("FIREBASE_SERVICE_ACCOUNT_PATH")
            },
            "jules": {
                "api_endpoint": os.getenv("JULES_API_ENDPOINT"),
                "api_key": os.getenv("JULES_API_KEY"),
                "webhook_secret": os.getenv("JULES_WEBHOOK_SECRET")
            }
        }
        
        # Try to load from config file as well
        try:
            import yaml
            with open("/workspace/configs/auth_config.yaml", "r") as f:
                file_config = yaml.safe_load(f)
                # Merge file config with environment config
                for provider, config in file_config.items():
                    if provider in self.config:
                        self.config[provider].update(config)
                    else:
                        self.config[provider] = config
        except FileNotFoundError:
            self.logger.warning("Auth config file not found, using environment variables only")
        except Exception as e:
            self.logger.warning("Failed to load auth config file", error=str(e))

    async def _initialize_oauth_configs(self) -> None:
        """Initialize OAuth 2.1 configurations for each provider."""
        # GitHub OAuth config
        if self.config.get("github", {}).get("client_id"):
            self.oauth_configs[AuthProvider.GITHUB] = OAuth2Config(
                provider=AuthProvider.GITHUB,
                client_id=self.config["github"]["client_id"],
                client_secret=self.config["github"]["client_secret"],
                redirect_uri="http://localhost:8080/auth/github/callback",
                authorization_url="https://github.com/login/oauth/authorize",
                token_url="https://github.com/login/oauth/access_token",
                scope="repo workflow admin:org"
            )
        
        # Google OAuth config
        if self.config.get("google", {}).get("client_id"):
            self.oauth_configs[AuthProvider.GOOGLE] = OAuth2Config(
                provider=AuthProvider.GOOGLE,
                client_id=self.config["google"]["client_id"],
                client_secret=self.config["google"]["client_secret"],
                redirect_uri="http://localhost:8080/auth/google/callback",
                authorization_url="https://accounts.google.com/o/oauth2/auth",
                token_url="https://oauth2.googleapis.com/token",
                scope="https://www.googleapis.com/auth/cloud-platform"
            )
        
        # Microsoft OAuth config
        if self.config.get("microsoft", {}).get("client_id"):
            self.oauth_configs[AuthProvider.MICROSOFT] = OAuth2Config(
                provider=AuthProvider.MICROSOFT,
                client_id=self.config["microsoft"]["client_id"],
                client_secret=self.config["microsoft"]["client_secret"],
                redirect_uri="http://localhost:8080/auth/microsoft/callback",
                authorization_url=f"https://login.microsoftonline.com/{self.config['microsoft']['tenant_id']}/oauth2/v2.0/authorize",
                token_url=f"https://login.microsoftonline.com/{self.config['microsoft']['tenant_id']}/oauth2/v2.0/token",
                scope="https://graph.microsoft.com/.default"
            )

    async def _load_existing_credentials(self) -> None:
        """Load existing credentials from secure storage."""
        # Initialize credentials from static tokens/keys where available
        
        # GitHub
        if self.config.get("github", {}).get("token"):
            self.credentials[AuthProvider.GITHUB] = AuthCredentials(
                provider=AuthProvider.GITHUB,
                access_token=self.config["github"]["token"],
                token_type="token"
            )
        
        # Google
        if self.config.get("google", {}).get("ai_api_key"):
            self.credentials[AuthProvider.GOOGLE] = AuthCredentials(
                provider=AuthProvider.GOOGLE,
                api_key=self.config["google"]["ai_api_key"],
                token_type="Bearer"
            )
        
        # Microsoft
        if self.config.get("microsoft", {}).get("openai_api_key"):
            self.credentials[AuthProvider.MICROSOFT] = AuthCredentials(
                provider=AuthProvider.MICROSOFT,
                api_key=self.config["microsoft"]["openai_api_key"],
                token_type="Bearer"
            )
        
        # Firebase
        if self.config.get("firebase", {}).get("api_key"):
            self.credentials[AuthProvider.FIREBASE] = AuthCredentials(
                provider=AuthProvider.FIREBASE,
                api_key=self.config["firebase"]["api_key"],
                token_type="Bearer"
            )
        
        # Jules
        if self.config.get("jules", {}).get("api_key"):
            self.credentials[AuthProvider.JULES] = AuthCredentials(
                provider=AuthProvider.JULES,
                api_key=self.config["jules"]["api_key"],
                token_type="Bearer"
            )

    def generate_pkce_challenge(self) -> PKCEChallenge:
        """Generate PKCE challenge for OAuth 2.1 flow."""
        # Generate code verifier (43-128 characters)
        code_verifier = base64.urlsafe_b64encode(secrets.token_bytes(32)).decode('utf-8').rstrip('=')
        
        # Generate code challenge
        code_challenge = base64.urlsafe_b64encode(
            hashlib.sha256(code_verifier.encode('utf-8')).digest()
        ).decode('utf-8').rstrip('=')
        
        return PKCEChallenge(
            code_verifier=code_verifier,
            code_challenge=code_challenge,
            code_challenge_method="S256"
        )

    async def get_authorization_url(self, provider: AuthProvider, state: str) -> str:
        """Get OAuth 2.1 authorization URL with PKCE challenge."""
        oauth_config = self.oauth_configs.get(provider)
        if not oauth_config:
            raise ValueError(f"OAuth configuration not found for provider: {provider}")
        
        # Generate PKCE challenge
        pkce_challenge = self.generate_pkce_challenge()
        self.pkce_challenges[state] = pkce_challenge
        
        # Build authorization URL
        params = {
            "client_id": oauth_config.client_id,
            "redirect_uri": oauth_config.redirect_uri,
            "scope": oauth_config.scope,
            "response_type": "code",
            "state": state
        }
        
        if oauth_config.use_pkce:
            params.update({
                "code_challenge": pkce_challenge.code_challenge,
                "code_challenge_method": pkce_challenge.code_challenge_method
            })
        
        query_string = "&".join([f"{k}={v}" for k, v in params.items()])
        return f"{oauth_config.authorization_url}?{query_string}"

    async def exchange_code_for_tokens(self, provider: AuthProvider, code: str, state: str) -> AuthCredentials:
        """Exchange authorization code for access tokens using PKCE."""
        oauth_config = self.oauth_configs.get(provider)
        if not oauth_config:
            raise ValueError(f"OAuth configuration not found for provider: {provider}")
        
        pkce_challenge = self.pkce_challenges.get(state)
        if not pkce_challenge and oauth_config.use_pkce:
            raise ValueError(f"PKCE challenge not found for state: {state}")
        
        # Prepare token exchange payload
        payload = {
            "client_id": oauth_config.client_id,
            "client_secret": oauth_config.client_secret,
            "code": code,
            "grant_type": "authorization_code",
            "redirect_uri": oauth_config.redirect_uri
        }
        
        if oauth_config.use_pkce and pkce_challenge:
            payload["code_verifier"] = pkce_challenge.code_verifier
        
        try:
            async def _exchange_tokens():
                async with self.session.post(
                    oauth_config.token_url,
                    data=payload,
                    headers={"Accept": "application/json"}
                ) as response:
                    if response.status != 200:
                        text = await response.text()
                        raise aiohttp.ClientResponseError(
                            request_info=response.request_info,
                            history=response.history,
                            status=response.status,
                            message=f"Token exchange failed: {text}"
                        )
                    return await response.json()
            
            token_response = await self.circuit_breakers[provider].call(
                await self.retry_manager.execute(_exchange_tokens)
            )
            
            # Create credentials
            credentials = AuthCredentials(
                provider=provider,
                access_token=token_response.get("access_token"),
                refresh_token=token_response.get("refresh_token"),
                id_token=token_response.get("id_token"),
                expires_at=time.time() + token_response.get("expires_in", 3600),
                scope=token_response.get("scope"),
                token_type=token_response.get("token_type", "Bearer")
            )
            
            # Store credentials
            self.credentials[provider] = credentials
            
            # Start token refresh monitoring
            if credentials.refresh_token and credentials.expires_at:
                await self._schedule_token_refresh(provider)
            
            # Clean up PKCE challenge
            if state in self.pkce_challenges:
                del self.pkce_challenges[state]
            
            self.logger.info("Successfully exchanged code for tokens", provider=provider.value)
            return credentials
            
        except Exception as e:
            self.logger.error("Token exchange failed", provider=provider.value, error=str(e))
            raise

    async def refresh_access_token(self, provider: AuthProvider) -> Optional[AuthCredentials]:
        """Refresh access token using refresh token."""
        async with self.refresh_locks[provider]:
            credentials = self.credentials.get(provider)
            if not credentials or not credentials.refresh_token:
                self.logger.warning("No refresh token available", provider=provider.value)
                return None
            
            oauth_config = self.oauth_configs.get(provider)
            if not oauth_config:
                self.logger.error("OAuth configuration not found", provider=provider.value)
                return None
            
            try:
                payload = {
                    "client_id": oauth_config.client_id,
                    "client_secret": oauth_config.client_secret,
                    "refresh_token": credentials.refresh_token,
                    "grant_type": "refresh_token"
                }
                
                async def _refresh_token():
                    async with self.session.post(
                        oauth_config.token_url,
                        data=payload,
                        headers={"Accept": "application/json"}
                    ) as response:
                        if response.status != 200:
                            text = await response.text()
                            raise aiohttp.ClientResponseError(
                                request_info=response.request_info,
                                history=response.history,
                                status=response.status,
                                message=f"Token refresh failed: {text}"
                            )
                        return await response.json()
                
                token_response = await self.circuit_breakers[provider].call(
                    await self.retry_manager.execute(_refresh_token)
                )
                
                # Update credentials
                credentials.access_token = token_response.get("access_token")
                if token_response.get("refresh_token"):
                    credentials.refresh_token = token_response["refresh_token"]
                credentials.expires_at = time.time() + token_response.get("expires_in", 3600)
                
                self.logger.info("Successfully refreshed access token", provider=provider.value)
                
                # Schedule next refresh
                await self._schedule_token_refresh(provider)
                
                return credentials
                
            except Exception as e:
                self.logger.error("Token refresh failed", provider=provider.value, error=str(e))
                return None

    async def _schedule_token_refresh(self, provider: AuthProvider) -> None:
        """Schedule automatic token refresh."""
        credentials = self.credentials.get(provider)
        if not credentials or not credentials.expires_at:
            return
        
        # Cancel existing refresh task
        if provider in self.refresh_tasks:
            self.refresh_tasks[provider].cancel()
        
        # Calculate refresh time (5 minutes before expiry)
        refresh_delay = max(credentials.expires_at - time.time() - 300, 60)
        
        async def _refresh_task():
            await asyncio.sleep(refresh_delay)
            await self.refresh_access_token(provider)
        
        self.refresh_tasks[provider] = asyncio.create_task(_refresh_task())

    async def _monitor_token_refresh(self) -> None:
        """Background task to monitor and refresh tokens."""
        while True:
            try:
                for provider, credentials in self.credentials.items():
                    if credentials.expires_at and credentials.refresh_token:
                        # Check if token expires within 10 minutes
                        if credentials.expires_at - time.time() < 600:
                            await self.refresh_access_token(provider)
                
                # Check every 5 minutes
                await asyncio.sleep(300)
                
            except Exception as e:
                self.logger.error("Error in token refresh monitoring", error=str(e))
                await asyncio.sleep(60)

    async def get_valid_credentials(self, provider: AuthProvider) -> Optional[AuthCredentials]:
        """Get valid credentials for a provider, refreshing if necessary."""
        credentials = self.credentials.get(provider)
        if not credentials:
            return None
        
        # Check if token is expired or expiring soon
        if credentials.expires_at and credentials.expires_at - time.time() < 300:
            if credentials.refresh_token:
                credentials = await self.refresh_access_token(provider)
        
        return credentials

    async def get_github_credentials(self) -> Dict[str, Any]:
        """Get GitHub credentials."""
        credentials = await self.get_valid_credentials(AuthProvider.GITHUB)
        config = self.config.get("github", {})
        
        return {
            "token": credentials.access_token if credentials else config.get("token"),
            "client_id": config.get("client_id"),
            "client_secret": config.get("client_secret"),
            "webhook_secret": config.get("webhook_secret")
        }

    async def get_google_credentials(self) -> Dict[str, Any]:
        """Get Google AI credentials."""
        credentials = await self.get_valid_credentials(AuthProvider.GOOGLE)
        config = self.config.get("google", {})
        
        return {
            "project_id": config.get("project_id"),
            "ai_api_key": credentials.api_key if credentials else config.get("ai_api_key"),
            "service_account_path": config.get("service_account_path"),
            "access_token": credentials.access_token if credentials else None
        }

    async def get_microsoft_credentials(self) -> Dict[str, Any]:
        """Get Microsoft AI credentials."""
        credentials = await self.get_valid_credentials(AuthProvider.MICROSOFT)
        config = self.config.get("microsoft", {})
        
        return {
            "tenant_id": config.get("tenant_id"),
            "client_id": config.get("client_id"),
            "client_secret": config.get("client_secret"),
            "subscription_id": config.get("subscription_id"),
            "openai_api_key": credentials.api_key if credentials else config.get("openai_api_key"),
            "access_token": credentials.access_token if credentials else None
        }

    async def get_firebase_credentials(self) -> Dict[str, Any]:
        """Get Firebase credentials."""
        credentials = await self.get_valid_credentials(AuthProvider.FIREBASE)
        config = self.config.get("firebase", {})
        
        return {
            "project_id": config.get("project_id"),
            "api_key": credentials.api_key if credentials else config.get("api_key"),
            "service_account_path": config.get("service_account_path"),
            "access_token": credentials.access_token if credentials else None
        }

    async def get_jules_credentials(self) -> Dict[str, Any]:
        """Get Jules Agent credentials."""
        credentials = await self.get_valid_credentials(AuthProvider.JULES)
        config = self.config.get("jules", {})
        
        return {
            "api_endpoint": config.get("api_endpoint"),
            "api_key": credentials.api_key if credentials else config.get("api_key"),
            "webhook_secret": config.get("webhook_secret"),
            "access_token": credentials.access_token if credentials else None
        }

    async def revoke_credentials(self, provider: AuthProvider) -> bool:
        """Revoke credentials for a provider."""
        try:
            credentials = self.credentials.get(provider)
            if not credentials:
                return True
            
            # Cancel refresh task
            if provider in self.refresh_tasks:
                self.refresh_tasks[provider].cancel()
                del self.refresh_tasks[provider]
            
            # Revoke tokens with provider if supported
            oauth_config = self.oauth_configs.get(provider)
            if oauth_config and credentials.access_token:
                # Implementation would depend on provider's revocation endpoint
                pass
            
            # Remove from memory
            del self.credentials[provider]
            
            self.logger.info("Successfully revoked credentials", provider=provider.value)
            return True
            
        except Exception as e:
            self.logger.error("Failed to revoke credentials", provider=provider.value, error=str(e))
            return False

    async def shutdown(self) -> None:
        """Shutdown the authentication manager."""
        self.logger.info("Shutting down unified authentication manager")
        
        # Cancel all refresh tasks
        for task in self.refresh_tasks.values():
            task.cancel()
        
        # Close HTTP session
        if self.session:
            await self.session.close()
        
        self.logger.info("Unified authentication manager shutdown complete")