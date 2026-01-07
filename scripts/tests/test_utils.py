"""
Tests for shared utilities.

Run with: pytest tests/test_utils.py -v
"""

import pytest
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.validators import (
    validate_url,
    validate_file_path,
    validate_api_key,
    validate_temperature,
    sanitize_filename,
    ValidationError
)


class TestValidators:
    """Test validation functions."""
    
    def test_validate_url_valid(self):
        """Test valid URL validation."""
        url = "https://api.example.com"
        assert validate_url(url) == url
    
    def test_validate_url_invalid(self):
        """Test invalid URL validation."""
        with pytest.raises(ValidationError):
            validate_url("not a url")
    
    def test_validate_url_requires_https(self):
        """Test HTTPS requirement."""
        with pytest.raises(ValidationError):
            validate_url("http://api.example.com", require_https=True)
    
   def test_validate_api_key_valid(self):
        """Test valid API key."""
        key = "sk-1234567890abcdef1234567890abcdef"
        assert validate_api_key(key) == key
    
    def test_validate_api_key_too_short(self):
        """Test API key too short."""
        with pytest.raises(ValidationError):
            validate_api_key("short")
    
    def test_validate_api_key_placeholder(self):
        """Test placeholder API key detection."""
        with pytest.raises(ValidationError):
            validate_api_key("your_api_key_here_xxx")
    
    def test_validate_temperature_valid(self):
        """Test valid temperature."""
        assert validate_temperature(0.7) == 0.7
        assert validate_temperature(0.0) == 0.0
        assert validate_temperature(2.0) == 2.0
    
    def test_validate_temperature_invalid(self):
        """Test invalid temperature."""
        with pytest.raises(ValidationError):
            validate_temperature(3.0)
        with pytest.raises(ValidationError):
            validate_temperature(-0.1)
    
    def test_sanitize_filename(self):
        """Test filename sanitization."""
        assert sanitize_filename("test.txt") == "test.txt"
        assert sanitize_filename("test/../../etc/passwd") == "test___etc_passwd"
        assert sanitize_filename("file:test?.txt") == "file_test_.txt"


class TestFilePathValidation:
    """Test file path validation."""
    
    def test_validate_existing_file(self, tmp_path):
        """Test validation of existing file."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("test")
        
        result = validate_file_path(str(test_file), must_exist=True)
        assert result.exists()
    
    def test_validate_nonexistent_file(self):
        """Test validation of nonexistent file."""
        with pytest.raises(ValidationError):
            validate_file_path("/nonexistent/path.txt", must_exist=True)
    
    def test_create_directory_if_missing(self, tmp_path):
        """Test directory creation."""
        new_dir = tmp_path / "new" / "nested" / "dir"
        result = validate_file_path(str(new_dir), create_if_missing=True)
        assert result.exists()
        assert result.is_dir()


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
