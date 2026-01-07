"""
Input validation utilities for LLM red team scripts.

Provides common validation functions for URLs, file paths, API keys,
and other user inputs.
"""

import re
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse
import ipaddress


class ValidationError(Exception):
    """Raised when validation fails."""
    pass


def validate_url(url: str, require_https: bool = False) -> str:
    """
    Validate and normalize a URL.
    
    Args:
        url: URL string to validate
        require_https: If True, only accept HTTPS URLs
        
    Returns:
        Normalized URL string
        
    Raises:
        ValidationError: If URL is invalid
        
    Example:
        >>> validate_url("https://api.example.com")
        'https://api.example.com'
        >>> validate_url("http://api.example.com", require_https=True)
        ValidationError: URL must use HTTPS
    """
    try:
        parsed = urlparse(url)
        
        if not parsed.scheme:
            raise ValidationError("URL must include scheme (http:// or https://)")
        
        if not parsed.netloc:
            raise ValidationError("URL must include domain")
        
        if require_https and parsed.scheme != 'https':
            raise ValidationError("URL must use HTTPS")
        
        return url
        
    except ValueError as e:
        raise ValidationError(f"Invalid URL: {e}")


def validate_file_path(
    path: str,
    must_exist: bool = False,
    must_be_file: bool = False,
    must_be_dir: bool = False,
    create_if_missing: bool = False
) -> Path:
    """
    Validate a file path.
    
    Args:
        path: Path string to validate
        must_exist: If True, path must exist
        must_be_file: If True, path must be a file
        must_be_dir: If True, path must be a directory
        create_if_missing: If True, create directory if missing
        
    Returns:
        Path object
        
    Raises:
        ValidationError: If validation fails
    """
    try:
        p = Path(path).resolve()
        
        if must_exist and not p.exists():
            raise ValidationError(f"Path does not exist: {path}")
        
        if must_be_file and p.exists() and not p.is_file():
            raise ValidationError(f"Path is not a file: {path}")
        
        if must_be_dir and p.exists() and not p.is_dir():
            raise ValidationError(f"Path is not a directory: {path}")
        
        if create_if_missing and not p.exists():
            p.mkdir(parents=True)
        
        return p
        
    except (OSError, RuntimeError) as e:
        raise ValidationError(f"Invalid path: {e}")


def validate_api_key(key: str, min_length: int = 20) -> str:
    """
    Validate an API key format.
    
    Args:
        key: API key string
        min_length: Minimum expected key length
        
    Returns:
        The API key if valid
        
    Raises:
        ValidationError: If key format is invalid
    """
    if not key or not key.strip():
        raise ValidationError("API key cannot be empty")
    
    key = key.strip()
    
    if len(key) < min_length:
        raise ValidationError(f"API key too short (minimum {min_length} characters)")
    
    # Check for obviously fake/placeholder keys
    placeholder_patterns = ['xxx', 'yyy', 'test', 'sample', 'placeholder', 'your_key_here']
    if any(pattern in key.lower() for pattern in placeholder_patterns):
        raise ValidationError("API key appears to be a placeholder")
    
    return key


def validate_ip_address(ip: str, allow_private: bool = True) -> str:
    """
    Validate an IP address.
    
    Args:
        ip: IP address string
        allow_private: If False, reject private IP ranges
        
    Returns:
        Normalized IP address
        
    Raises:
        ValidationError: If IP is invalid
    """
    try:
        ip_obj = ipaddress.ip_address(ip)
        
        if not allow_private and ip_obj.is_private:
            raise ValidationError(f"Private IP addresses not allowed: {ip}")
        
        return str(ip_obj)
        
    except ValueError as e:
        raise ValidationError(f"Invalid IP address: {e}")


def validate_port(port: int) -> int:
    """
    Validate a port number.
    
    Args:
        port: Port number to validate
        
    Returns:
        The port number if valid
        
    Raises:
        ValidationError: If port is out of range
    """
    if not isinstance(port, int):
        try:
            port = int(port)
        except (ValueError, TypeError):
            raise ValidationError(f"Port must be an integer: {port}")
    
    if port < 1 or port > 65535:
        raise ValidationError(f"Port must be between 1 and 65535: {port}")
    
    return port


def validate_prompt(prompt: str, max_length: int = 100000) -> str:
    """
    Validate a prompt string.
    
    Args:
        prompt: Prompt text to validate
        max_length: Maximum allowed length
        
    Returns:
        The prompt if valid
        
    Raises:
        ValidationError: If prompt is invalid
    """
    if not prompt or not prompt.strip():
        raise ValidationError("Prompt cannot be empty")
    
    if len(prompt) > max_length:
        raise ValidationError(f"Prompt exceeds maximum length ({max_length} characters)")
    
    return prompt


def validate_temperature(temp: float) -> float:
    """
    Validate LLM temperature parameter.
    
    Args:
        temp: Temperature value to validate
        
    Returns:
        The temperature if valid
        
    Raises:
        ValidationError: If temperature out of range
    """
    try:
        temp = float(temp)
    except (ValueError, TypeError):
        raise ValidationError(f"Temperature must be a number: {temp}")
    
    if temp < 0.0 or temp > 2.0:
        raise ValidationError(f"Temperature must be between 0.0 and 2.0: {temp}")
    
    return temp


def sanitize_filename(filename: str, max_length: int = 255) -> str:
    """
    Sanitize a filename by removing dangerous characters.
    
    Args:
        filename: Original filename
        max_length: Maximum filename length
        
    Returns:
        Sanitized filename
        
    Example:
        >>> sanitize_filename("test/../../etc/passwd")
        'test___etc_passwd'
    """
    # Remove path separators and dangerous characters
    sanitized = re.sub(r'[/\\:*?"<>|]', '_', filename)
    
    # Remove leading dots (hidden files)
    sanitized = sanitized.lstrip('.')
    
    # Limit length
    if len(sanitized) > max_length:
        sanitized = sanitized[:max_length]
    
    return sanitized or 'unnamed'
