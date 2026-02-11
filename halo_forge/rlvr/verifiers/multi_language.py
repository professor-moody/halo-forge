"""
Multi-Language Verifier with Auto-Detection

Automatically detects the programming language from code patterns
and routes to the appropriate verifier.

Supports:
- C/C++ (GCC, MinGW, Clang)
- Python (pytest)
- Rust (cargo)
- Go (go build)
- C# (.NET)
- PowerShell

Usage:
    verifier = MultiLanguageVerifier()
    result = verifier.verify(code)  # Auto-detects language
    
    # Or with explicit language
    result = verifier.verify(code, language="rust")
"""

import re
import inspect
from typing import Optional, Dict, List, Any
from dataclasses import dataclass, field

from halo_forge.rlvr.verifiers.base import Verifier, VerifyResult


@dataclass
class LanguageConfig:
    """Configuration for a supported language."""
    name: str
    patterns: List[str]           # Regex patterns to detect this language
    verifier_class: str           # Class name of verifier to use
    verifier_kwargs: Dict = field(default_factory=dict)
    priority: int = 0             # Higher = check first


# Default language patterns
DEFAULT_LANGUAGE_CONFIGS = {
    'cpp': LanguageConfig(
        name='cpp',
        patterns=[
            r'#include\s*<(?:iostream|cstdio|vector|string|algorithm|map|set|queue)',
            r'#include\s*<windows\.h>',
            r'int\s+main\s*\(\s*(?:int\s+argc|void)?\s*',
            r'std::\w+',
            r'using\s+namespace\s+std',
            r'cout\s*<<',
            r'printf\s*\(',
        ],
        verifier_class='GCCVerifier',
        priority=10
    ),
    'c': LanguageConfig(
        name='c',
        patterns=[
            r'#include\s*<stdio\.h>',
            r'#include\s*<stdlib\.h>',
            r'#include\s*<string\.h>',
        ],
        verifier_class='GCCVerifier',
        verifier_kwargs={'flags': ['-O2', '-Wall']},
        priority=5
    ),
    'python': LanguageConfig(
        name='python',
        patterns=[
            r'^def\s+\w+\s*\(',
            r'^class\s+\w+\s*[\(:]',
            r'^import\s+\w+',
            r'^from\s+\w+\s+import',
            r'print\s*\(',
            r'if\s+__name__\s*==\s*[\'"]__main__[\'"]:',
        ],
        verifier_class='PytestVerifier',
        priority=8
    ),
    'rust': LanguageConfig(
        name='rust',
        patterns=[
            r'^fn\s+main\s*\(\s*\)',
            r'^use\s+std::',
            r'^mod\s+\w+',
            r'println!\s*\(',
            r'let\s+mut\s+\w+',
            r'->\s*(?:Result|Option|Vec|String)',
        ],
        verifier_class='RustVerifier',
        priority=9
    ),
    'go': LanguageConfig(
        name='go',
        patterns=[
            r'^package\s+main',
            r'^func\s+main\s*\(\s*\)',
            r'^import\s+\(',
            r'fmt\.Println\(',
            r':=\s*',
        ],
        verifier_class='GoVerifier',
        priority=9
    ),
    'csharp': LanguageConfig(
        name='csharp',
        patterns=[
            r'^using\s+System',
            r'^namespace\s+\w+',
            r'class\s+\w+\s*(?::\s*\w+)?',
            r'static\s+void\s+Main\s*\(',
            r'Console\.WriteLine\s*\(',
        ],
        verifier_class='DotNetVerifier',
        priority=7
    ),
    'powershell': LanguageConfig(
        name='powershell',
        patterns=[
            r'\$\w+\s*=',
            r'Write-Host\s+',
            r'Get-\w+',
            r'Set-\w+',
            r'\[Parameter\s*\(',
            r'function\s+\w+\s*{',
        ],
        verifier_class='PowerShellVerifier',
        priority=6
    ),
}


class MultiLanguageVerifier(Verifier):
    """
    Auto-detecting multi-language verifier.
    
    Detects the programming language from code patterns and routes
    to the appropriate language-specific verifier.
    
    Example:
        verifier = MultiLanguageVerifier()
        
        # Auto-detect language
        result = verifier.verify(cpp_code)   # Uses GCCVerifier
        result = verifier.verify(python_code) # Uses PytestVerifier
        
        # Explicit language
        result = verifier.verify(code, language='rust')
        
        # With execution tests
        verifier = MultiLanguageVerifier(run_after_compile=True)
    """
    
    def __init__(
        self,
        language_configs: Optional[Dict[str, LanguageConfig]] = None,
        default_language: str = 'python',
        max_workers: int = 8,
        run_after_compile: bool = False,
        binary_cache_dir: Optional[str] = None,
        **verifier_kwargs
    ):
        """
        Initialize multi-language verifier.
        
        Args:
            language_configs: Custom language configurations
            default_language: Fallback language if detection fails
            max_workers: Max parallel workers
            run_after_compile: Enable execution for compile verifiers
            binary_cache_dir: Directory to cache compiled binaries
            **verifier_kwargs: Additional kwargs passed to all verifiers
        """
        super().__init__(max_workers=max_workers)
        
        self.language_configs = language_configs or DEFAULT_LANGUAGE_CONFIGS.copy()
        self.default_language = default_language
        self.run_after_compile = run_after_compile
        self.binary_cache_dir = binary_cache_dir
        self.verifier_kwargs = verifier_kwargs
        
        # Lazy-loaded verifiers
        self._verifiers: Dict[str, Verifier] = {}
        self._verifier_init_metadata: Dict[str, Dict[str, Any]] = {}
    
    def _get_verifier(self, language: str) -> Verifier:
        """Get or create verifier for a language."""
        if language not in self._verifiers:
            config = self.language_configs.get(language)
            if not config:
                raise ValueError(f"Unknown language: {language}")
            
            # Import and instantiate the verifier
            verifier, init_metadata = self._create_verifier(config)
            self._verifiers[language] = verifier
            self._verifier_init_metadata[language] = init_metadata
        
        return self._verifiers[language]
    
    def _create_verifier(self, config: LanguageConfig) -> tuple[Verifier, Dict[str, Any]]:
        """Create a verifier instance from config."""
        from halo_forge.rlvr import verifiers
        
        # Get the verifier class
        verifier_class = getattr(verifiers, config.verifier_class, None)
        if not verifier_class:
            raise ValueError(f"Unknown verifier class: {config.verifier_class}")
        
        # Merge kwargs
        kwargs = {
            'max_workers': self.max_workers,
            **config.verifier_kwargs,
            **self.verifier_kwargs
        }

        applied_optional_kwargs: Dict[str, Any] = {}
        skipped_optional_kwargs: Dict[str, str] = {}

        try:
            signature = inspect.signature(verifier_class.__init__)
            params = signature.parameters
            accepts_var_kwargs = any(
                p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()
            )
        except (TypeError, ValueError):
            params = {}
            accepts_var_kwargs = True

        def _supports_kwarg(name: str) -> bool:
            return accepts_var_kwargs or name in params

        if _supports_kwarg('run_after_compile'):
            kwargs['run_after_compile'] = self.run_after_compile
            applied_optional_kwargs['run_after_compile'] = self.run_after_compile
        else:
            skipped_optional_kwargs['run_after_compile'] = (
                f"{config.verifier_class}.__init__ does not accept run_after_compile"
            )

        if self.binary_cache_dir:
            if _supports_kwarg('binary_cache_dir'):
                kwargs['binary_cache_dir'] = self.binary_cache_dir
                applied_optional_kwargs['binary_cache_dir'] = self.binary_cache_dir
            else:
                skipped_optional_kwargs['binary_cache_dir'] = (
                    f"{config.verifier_class}.__init__ does not accept binary_cache_dir"
                )

        verifier = verifier_class(**kwargs)
        init_metadata = {
            'verifier_class': config.verifier_class,
            'applied_optional_kwargs': applied_optional_kwargs,
            'skipped_optional_kwargs': skipped_optional_kwargs,
        }
        return verifier, init_metadata
    
    def detect_language(self, code: str) -> str:
        """
        Detect the programming language from code.
        
        Args:
            code: Source code to analyze
            
        Returns:
            Detected language name
        """
        # Sort by priority (descending)
        sorted_configs = sorted(
            self.language_configs.items(),
            key=lambda x: x[1].priority,
            reverse=True
        )
        
        # Check each language's patterns
        for lang, config in sorted_configs:
            for pattern in config.patterns:
                if re.search(pattern, code, re.MULTILINE):
                    return lang
        
        return self.default_language
    
    def verify(self, code: str, language: Optional[str] = None) -> VerifyResult:
        """
        Verify code using auto-detected or specified language.
        
        Args:
            code: Source code to verify
            language: Optional language override
            
        Returns:
            VerifyResult from language-specific verifier
        """
        # Extract code if needed
        extracted = self.extract_code(code)
        
        # Detect or use provided language
        lang = language or self.detect_language(extracted)
        
        try:
            verifier = self._get_verifier(lang)
            result = verifier.verify(extracted)
            
            # Add language info to metadata
            if result.metadata:
                result.metadata['detected_language'] = lang
            else:
                result.metadata = {'detected_language': lang}
            result.metadata['verifier_name'] = verifier.__class__.__name__
            result.metadata['verifier_init'] = self._verifier_init_metadata.get(
                lang,
                {},
            )
            
            return result
            
        except Exception as e:
            return VerifyResult(
                success=False,
                reward=0.0,
                details=f"Verification failed for language '{lang}'",
                error=str(e),
                metadata={'detected_language': lang}
            )
    
    def verify_batch(
        self,
        codes: List[str],
        prompts: Optional[List[str]] = None,
        languages: Optional[List[str]] = None
    ) -> List[VerifyResult]:
        """
        Verify multiple code samples, potentially in different languages.
        
        Args:
            codes: List of code strings
            prompts: Optional prompts (ignored, for API compatibility)
            languages: Optional per-code language overrides
            
        Returns:
            List of VerifyResult objects
        """
        if languages and len(languages) != len(codes):
            raise ValueError("languages must have same length as codes")
        
        results = []
        for i, code in enumerate(codes):
            lang = languages[i] if languages else None
            results.append(self.verify(code, language=lang))
        
        return results
    
    def cleanup(self):
        """Cleanup all verifiers."""
        for verifier in self._verifiers.values():
            verifier.cleanup()
        self._verifiers.clear()
    
    @property
    def supported_languages(self) -> List[str]:
        """Get list of supported languages."""
        return list(self.language_configs.keys())


class AutoVerifier(MultiLanguageVerifier):
    """Alias for MultiLanguageVerifier for CLI convenience."""
    pass
