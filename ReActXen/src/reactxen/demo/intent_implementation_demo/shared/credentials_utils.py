"""
Utility functions for managing credentials.
Loads credentials from .env and syncs to credentials.json.
"""
import json
import os
from pathlib import Path
from dotenv import load_dotenv
from typing import Dict, Optional

# Load environment variables
load_dotenv(override=False)

# Paths
CREDENTIALS_TEMPLATE = Path(__file__).parent.parent / "credentials.json.template"
CREDENTIALS_FILE = Path(__file__).parent.parent / "credentials.json"


def load_credentials_from_env() -> Dict[str, str]:
    """Load credentials from environment variables."""
    return {
        "watsonx_apikey": os.environ.get("WATSONX_APIKEY", ""),
        "watsonx_url": os.environ.get("WATSONX_URL", "https://us-south.ml.cloud.ibm.com/"),
        "watsonx_project_id": os.environ.get("WATSONX_PROJECT_ID", ""),
        "openai_api_key": os.environ.get("OPENAI_API_KEY", ""),
        "hf_api_key": os.environ.get("HF_API_KEY") or os.environ.get("HUGGINGFACE_API_KEY", ""),
        "brave_api_key": os.environ.get("BRAVE_API_KEY", ""),
    }


def sync_credentials_to_json() -> Path:
    """Sync credentials from .env to credentials.json."""
    credentials = load_credentials_from_env()
    
    # Load existing credentials.json if it exists to preserve any manual overrides
    if CREDENTIALS_FILE.exists():
        try:
            with open(CREDENTIALS_FILE, 'r') as f:
                existing = json.load(f)
                # Update with env vars, but preserve existing values if env is empty
                for key, value in credentials.items():
                    if value:  # Only update if env var has a value
                        existing[key] = value
                credentials = existing
        except json.JSONDecodeError:
            pass  # If file is corrupted, use env vars
    
    # Write credentials.json
    with open(CREDENTIALS_FILE, 'w') as f:
        json.dump(credentials, f, indent=2)
    
    return CREDENTIALS_FILE


def load_credentials() -> Dict[str, str]:
    """Load credentials from credentials.json, falling back to .env if needed."""
    # First, sync credentials from .env
    sync_credentials_to_json()
    
    # Load from credentials.json
    if CREDENTIALS_FILE.exists():
        try:
            with open(CREDENTIALS_FILE, 'r') as f:
                creds = json.load(f)
                # Fill in any missing values from env
                env_creds = load_credentials_from_env()
                for key, value in env_creds.items():
                    if key not in creds or not creds[key]:
                        creds[key] = value
                return creds
        except json.JSONDecodeError:
            pass
    
    # Fallback to environment variables
    return load_credentials_from_env()


# Auto-sync on import
sync_credentials_to_json()

