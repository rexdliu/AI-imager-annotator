"""Environment variable utilities for loading and accessing configuration."""

import os
from typing import Optional
from pathlib import Path

from dotenv import load_dotenv

# Load .env file from project root
env_path = Path(".env")
load_dotenv(dotenv_path=env_path)


def get_env_var(name: str, default: Optional[str] = None, required: bool = False) -> Optional[str]:
    """
    Get environment variable with options for defaults and required flags.

    Args:
        name: Name of the environment variable
        default: Default value if not found
        required: Whether the variable is required

    Returns:
        Value of the environment variable or default

    Raises:
        ValueError: If required is True and the variable is not set
    """
    value = os.getenv(name, default)
    if required and not value:
        raise ValueError(f"Required environment variable '{name}' is not set.")
    return value