# config.py
from pydantic_settings import BaseSettings
from pydantic import Field, ConfigDict
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from functools import lru_cache
import hashlib
import json
from pathlib import Path
from typing import Optional, Dict, Any

# Load .env file
load_dotenv()

class Settings(BaseSettings):
    """Application settings with validation and caching support."""
    
    # Required
    GOOGLE_API_KEY: str
    
    # Model Configuration - Optimized for cost/performance
    MODEL_NAME: str = "gemini-2.5-flash"  # Changed to faster, cheaper model
    TEMPERATURE: float = 0.1
    MAX_TOKENS_DOCSTRING: int = 400  # Reduced from 1024
    MAX_TOKENS_README: int = 1500    # Optimized for README
    
    # Safety Limits
    MAX_FILE_SIZE: int = 100_000      # 100 KB
    MAX_FILES_TOTAL: int = 500        # Max files to process
    MAX_DEPTH: int = 10              # Max directory depth
    MAX_ZIP_SIZE: int = 50 * 1024 * 1024  # 50MB
    MAX_DEPENDENCIES_SHOWN: int = 15  # Reduced from 30
    MAX_FUNCTIONS_SHOWN: int = 15     # Reduced from 20
    MAX_CLASSES_SHOWN: int = 8        # Reduced from 10
    
    # Cache Settings
    CACHE_DIR: Path = Field(default_factory=lambda: Path.home() / ".cache" / "gemini-doc-gen")
    CACHE_TTL_HOURS: int = 24
    ENABLE_CACHE: bool = True
    
    # Security
    ALLOWED_SCHEMES: set = {"file"}  # Only local filesystem
    DISALLOWED_PATHS: list = ["/etc", "/var", "/root", "C:\\Windows", "C:\\Program Files"]
    
    model_config = ConfigDict(
        env_file=".env",
        extra="ignore",
        frozen=True  # Immutable settings
    )


@lru_cache()
def get_settings() -> Settings:
    """Cached settings singleton."""
    return Settings()


# ============ Shared LLM Instance ============
@lru_cache(maxsize=1)
def get_llm(max_tokens: int = 400, temperature: float = None) -> ChatGoogleGenerativeAI:
    """
    Get cached LLM instance with configurable parameters.
    This ensures we reuse the same instance across the application.
    """
    settings = get_settings()
    
    return ChatGoogleGenerativeAI(
        model=settings.MODEL_NAME,
        google_api_key=settings.GOOGLE_API_KEY,
        temperature=temperature if temperature is not None else settings.TEMPERATURE,
        max_tokens=max_tokens,
        convert_system_message_to_human=True,
        timeout=30,  # Add timeout to prevent hanging
        max_retries=2  # Automatic retries on failure
    )


# ============ Cache Manager ============
class CacheManager:
    """File-based caching with TTL and hash-based keys."""
    
    def __init__(self):
        self.settings = get_settings()
        self.cache_dir = self.settings.CACHE_DIR
        self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    def _get_cache_key(self, data: Dict[str, Any]) -> str:
        """Generate stable hash from data."""
        # Sort keys for consistent hashing
        serialized = json.dumps(data, sort_keys=True)
        return hashlib.sha256(serialized.encode()).hexdigest()
    
    def get(self, key_data: Dict[str, Any]) -> Optional[str]:
        """Retrieve cached item if not expired."""
        if not self.settings.ENABLE_CACHE:
            return None
        
        cache_key = self._get_cache_key(key_data)
        cache_file = self.cache_dir / f"{cache_key}.json"
        
        if not cache_file.exists():
            return None
        
        try:
            with open(cache_file, 'r', encoding='utf-8') as f:
                cached = json.load(f)
            
            # Check TTL
            import time
            if time.time() - cached['timestamp'] < self.settings.CACHE_TTL_HOURS * 3600:
                return cached['value']
        except Exception:
            # If any error reading cache, ignore and regenerate
            pass
        
        return None
    
    def set(self, key_data: Dict[str, Any], value: str) -> None:
        """Store item in cache with timestamp."""
        if not self.settings.ENABLE_CACHE:
            return
        
        try:
            cache_key = self._get_cache_key(key_data)
            cache_file = self.cache_dir / f"{cache_key}.json"
            
            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump({
                    'timestamp': __import__('time').time(),
                    'value': value
                }, f)
        except Exception:
            # Cache failures are non-critical
            pass
    
    def clear_expired(self) -> int:
        """Clear expired cache entries. Returns count cleared."""
        cleared = 0
        import time
        now = time.time()
        
        for cache_file in self.cache_dir.glob("*.json"):
            try:
                with open(cache_file, 'r', encoding='utf-8') as f:
                    cached = json.load(f)
                
                if now - cached['timestamp'] > self.settings.CACHE_TTL_HOURS * 3600:
                    cache_file.unlink()
                    cleared += 1
            except Exception:
                # Delete corrupted cache files
                try:
                    cache_file.unlink()
                    cleared += 1
                except:
                    pass
        
        return cleared


# Singleton instances
settings = get_settings()
cache_manager = CacheManager()

# Validate critical settings
if not settings.GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY environment variable is not set")