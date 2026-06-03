#!/usr/bin/env python3
"""
Path Security Validator - Secure Path Sanitization
Handles path sanitization and security checks to prevent path traversal attacks.
"""

import os
import re
from pathlib import Path
from typing import Optional

from src.core.logging.logger import ProjectLogger

logger = ProjectLogger.get_logger("PathSecurityValidator")


class PathSecurityValidator:
    """
    Path security validator for secure path handling.
    
    Handles:
    - Path sanitization
    - Security checks
    - Traversal detection
    - Path bounds validation
    """
    MAX_PATH_LENGTH = 4096
    
    def __init__(self):
        """Initialize Path Security Validator."""
        self.logger = logger
        self.logger.info("✅ PathSecurityValidator initialized")
    
    def sanitize_path_input(self, path_input: str, base_dir: Optional[str] = None) -> str:
        """
        Secure path sanitization with multiple checks to prevent path traversal attacks.
        
        Args:
            path_input: Input path to sanitize
            base_dir: Base directory for bounds checking (optional)
            
        Returns:
            Sanitized path string
            
        Raises:
            ValueError: If path is empty or violates security constraints
        """
        if not path_input:
            raise ValueError("Path cannot be empty")
        
        # 1. Basic security checks (null bytes, normalization)
        sanitized = self._check_path_security_basics(path_input)
        
        # 2. Check for traversal and absolute paths
        self._check_path_traversal(sanitized)
        
        # 3. Handle base directory restriction
        if base_dir:
            return self._check_path_bounds(sanitized, base_dir)
        
        return sanitized
    
    def _check_path_security_basics(self, path_input: str) -> str:
        """Performs initial security normalization and cleaning."""
        if '\0' in path_input:
            raise ValueError("Null byte detected in path")
        
        # Remove control characters before normalization; reject abusive length
        # instead of truncating, because truncation can change security meaning.
        cleaned = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', path_input)
        normalized = os.path.normpath(cleaned)
        if len(normalized) > self.MAX_PATH_LENGTH:
            raise ValueError("Path is too long")
        return normalized
    
    def _check_path_traversal(self, path: str) -> None:
        """Validates that the path does not contain traversal or absolute markers."""
        parts = [part for part in re.split(r'[\\/]+', path) if part]
        if any(part == ".." for part in parts):
            raise ValueError("Path traversal detected (..)")
        
        if os.path.isabs(path):
            raise ValueError("Absolute paths not allowed")
        
        if re.match(r'^[A-Za-z]:', path):
            raise ValueError("Drive letters not allowed")
    
    def _check_path_bounds(self, path: str, base_dir: str) -> str:
        """Ensures the path remains within the specified base directory."""
        base_path = Path(base_dir).resolve()
        full_path = (base_path / path).resolve()
        
        try:
            full_path.relative_to(base_path)
        except ValueError as err:
            raise ValueError(f"Path outside allowed directory: {base_dir}") from err
        
        return str(full_path)
