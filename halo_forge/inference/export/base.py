"""
Base Model Exporter

Abstract base class for model exporters.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Dict, Any, List


@dataclass
class ExportConfig:
    """Configuration for model export."""
    output_path: str = "model"
    quantization: Optional[str] = None  # Q4_K_M, Q8_0, etc.
    optimize: bool = True
    metadata: Dict[str, str] = field(default_factory=dict)


class ModelExporter(ABC):
    """
    Abstract base class for model exporters.
    
    Implement this class to add new export formats.
    """
    
    def __init__(self, config: Optional[ExportConfig] = None):
        """
        Initialize exporter.
        
        Args:
            config: Export configuration
        """
        self.config = config or ExportConfig()
    
    @property
    @abstractmethod
    def format_name(self) -> str:
        """Return the export format name."""
        pass
    
    @property
    @abstractmethod
    def file_extension(self) -> str:
        """Return the file extension for this format."""
        pass
    
    @abstractmethod
    def export(
        self,
        model: Any,
        output_path: str,
        tokenizer: Any = None,
        **kwargs
    ) -> str:
        """
        Export model to this format.
        
        Args:
            model: Model to export
            output_path: Path to save exported model
            tokenizer: Optional tokenizer
            **kwargs: Format-specific options
            
        Returns:
            Path to exported model
        """
        pass
    
    def validate_model(self, model: Any) -> bool:
        """
        Validate that model can be exported.
        
        Args:
            model: Model to validate
            
        Returns:
            True if model can be exported
        """
        # Check if model has required attributes
        if not hasattr(model, 'config'):
            return False
        
        return True
    
    def get_model_info(self, model: Any) -> Dict[str, Any]:
        """
        Extract model information for export metadata.
        
        Args:
            model: Model to inspect
            
        Returns:
            Dictionary with model information
        """
        info = {
            "model_type": getattr(model.config, 'model_type', 'unknown'),
            "vocab_size": getattr(model.config, 'vocab_size', 0),
            "hidden_size": getattr(model.config, 'hidden_size', 0),
            "num_hidden_layers": getattr(model.config, 'num_hidden_layers', 0),
            "num_attention_heads": getattr(model.config, 'num_attention_heads', 0),
        }
        
        return info
    
    def prepare_output_path(self, output_path: str) -> Path:
        """
        Prepare output path, creating directories if needed.
        
        Args:
            output_path: Desired output path
            
        Returns:
            Resolved Path object
        """
        path = Path(output_path)
        
        # Add extension if not present
        if not path.suffix:
            path = path.with_suffix(self.file_extension)
        
        # Create parent directories
        path.parent.mkdir(parents=True, exist_ok=True)
        
        return path
