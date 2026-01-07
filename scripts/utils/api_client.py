"""
Shared API client utilities for LLM red team testing.

This module provides common functionality for interacting with LLM APIs,
including request handling, error management, and response parsing.
"""

import requests
import logging
from typing import Optional, Dict, Any, List
from dataclasses import dataclass
from enum import Enum
import time

logger = logging.getLogger(__name__)


class HTTPMethod(Enum):
    """Supported HTTP methods."""
    GET = "GET"
    POST = "POST"
    PUT = "PUT"
    DELETE = "DELETE"


@dataclass
class APIResponse:
    """Structured API response."""
    status_code: int
    data: Dict[str, Any]
    headers: Dict[str, str]
    elapsed_ms: float
    
    @property
    def success(self) -> bool:
        """Check if response was successful."""
        return 200 <= self.status_code < 300


class LLMAPIClient:
    """
    Generic client for LLM API interactions.
    
    Provides common functionality for making HTTP requests to LLM APIs
    with built-in retry logic, timeout handling, and error management.
    
    Example:
        >>> client = LLMAPIClient("https://api.example.com", api_key="key123")
        >>> response = client.send_prompt("Hello, AI!")
        >>> print(response.data)
    """
    
    def __init__(
        self,
        base_url: str,
        api_key: Optional[str] = None,
        timeout: int = 30,
        max_retries: int = 3
    ):
        """
        Initialize the API client.
        
        Args:
            base_url: Base URL for the API
            api_key: Optional API key for authentication
            timeout: Request timeout in seconds (default: 30)
            max_retries: Maximum number of retry attempts (default: 3)
        """
        self.base_url = base_url.rstrip('/')
        self.api_key = api_key
        self.timeout = timeout
        self.max_retries = max_retries
        self.session = requests.Session()
        
        if api_key:
            self.session.headers.update({'Authorization': f'Bearer {api_key}'})
    
    def _make_request(
        self,
        method: HTTPMethod,
        endpoint: str,
        data: Optional[Dict[str, Any]] = None,
        params: Optional[Dict[str, Any]] = None
    ) -> APIResponse:
        """
        Make an HTTP request with retry logic.
        
        Args:
            method: HTTP method to use
            endpoint: API endpoint (relative to base_url)
            data: Request body data
            params: Query parameters
            
        Returns:
            APIResponse object with structured response data
            
        Raises:
            requests.RequestException: If request fails after retries
        """
        url = f"{self.base_url}/{endpoint.lstrip('/')}"
        
        for attempt in range(self.max_retries):
            try:
                start_time = time.time()
                
                response = self.session.request(
                    method=method.value,
                    url=url,
                    json=data,
                    params=params,
                    timeout=self.timeout
                )
                
                elapsed_ms = (time.time() - start_time) * 1000
                
                return APIResponse(
                    status_code=response.status_code,
                    data=response.json() if response.content else {},
                    headers=dict(response.headers),
                    elapsed_ms=elapsed_ms
                )
                
            except requests.Timeout:
                logger.warning(f"Request timeout (attempt {attempt + 1}/{self.max_retries})")
                if attempt == self.max_retries - 1:
                    raise
                time.sleep(2 ** attempt)  # Exponential backoff
                
            except requests.RequestException as e:
                logger.error(f"Request failed: {e}")
                if attempt == self.max_retries - 1:
                    raise
                time.sleep(2 ** attempt)
    
    def send_prompt(
        self,
        prompt: str,
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 1000,
        **kwargs
    ) -> APIResponse:
        """
        Send a prompt to the LLM API.
        
        Args:
            prompt: The prompt text to send
            model: Optional model identifier
            temperature: Sampling temperature (0.0-1.0)
            max_tokens: Maximum tokens in response
            **kwargs: Additional parameters to pass to the API
            
        Returns:
            APIResponse with LLM completion
            
        Example:
            >>> client = LLMAPIClient("https://api.example.com")
            >>> resp = client.send_prompt("Explain quantum computing")
            >>> print(resp.data['choices'][0]['text'])
        """
        payload = {
            'prompt': prompt,
            'temperature': temperature,
            'max_tokens': max_tokens,
            **kwargs
        }
        
        if model:
            payload['model'] = model
        
        return self._make_request(HTTPMethod.POST, '/v1/completions', data=payload)
    
    def chat_completion(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        **kwargs
    ) -> APIResponse:
        """
        Send chat messages to the LLM API.
        
        Args:
            messages: List of message dicts with 'role' and 'content'
            model: Optional model identifier
            **kwargs: Additional parameters
            
        Returns:
            APIResponse with chat completion
            
        Example:
            >>> messages = [
            ...     {"role": "system", "content": "You are a helpful assistant"},
            ...     {"role": "user", "content": "Hello!"}
            ... ]
            >>> resp = client.chat_completion(messages)
        """
        payload = {
            'messages': messages,
            **kwargs
        }
        
        if model:
            payload['model'] = model
        
        return self._make_request(HTTPMethod.POST, '/v1/chat/completions', data=payload)
    
    def test_connection(self) -> bool:
        """
        Test if the API is reachable.
        
        Returns:
            True if connection successful, False otherwise
        """
        try:
            response = self._make_request(HTTPMethod.GET, '/health')
            return response.success
        except Exception as e:
            logger.error(f"Connection test failed: {e}")
            return False
