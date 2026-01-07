"""
Image Processors for VLM Models

Preprocessing pipelines for different VLM architectures.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Any, Optional, List, Tuple, Union
from pathlib import Path

import torch
from PIL import Image
import numpy as np


@dataclass
class ProcessedImage:
    """Processed image ready for VLM input."""
    pixel_values: torch.Tensor
    image_size: Tuple[int, int]
    original_size: Tuple[int, int]
    metadata: Dict[str, Any]


class ImageProcessor(ABC):
    """Abstract base class for image processors."""
    
    @abstractmethod
    def __call__(
        self,
        image: Union[Image.Image, str, Path],
        **kwargs
    ) -> ProcessedImage:
        """Process an image for VLM input."""
        pass
    
    def load_image(self, image: Union[Image.Image, str, Path]) -> Image.Image:
        """Load image from path if needed."""
        if isinstance(image, Image.Image):
            return image
        
        if isinstance(image, (str, Path)):
            return Image.open(image).convert('RGB')
        
        raise ValueError(f"Unknown image type: {type(image)}")
    
    def resize_image(
        self,
        image: Image.Image,
        target_size: Tuple[int, int],
        keep_aspect: bool = True
    ) -> Image.Image:
        """
        Resize image to target size.
        
        Args:
            image: PIL Image
            target_size: (width, height)
            keep_aspect: Maintain aspect ratio
            
        Returns:
            Resized image
        """
        if not keep_aspect:
            return image.resize(target_size, Image.LANCZOS)
        
        # Calculate size maintaining aspect ratio
        w, h = image.size
        target_w, target_h = target_size
        
        scale = min(target_w / w, target_h / h)
        new_w = int(w * scale)
        new_h = int(h * scale)
        
        resized = image.resize((new_w, new_h), Image.LANCZOS)
        
        # Pad to target size if needed
        if (new_w, new_h) != target_size:
            padded = Image.new('RGB', target_size, (0, 0, 0))
            paste_x = (target_w - new_w) // 2
            paste_y = (target_h - new_h) // 2
            padded.paste(resized, (paste_x, paste_y))
            return padded
        
        return resized


class VLMPreprocessor(ImageProcessor):
    """
    Generic VLM image preprocessor.
    
    Works with most vision-language models by using standard
    normalization and resizing.
    """
    
    def __init__(
        self,
        image_size: Tuple[int, int] = (336, 336),
        mean: Tuple[float, float, float] = (0.48145466, 0.4578275, 0.40821073),
        std: Tuple[float, float, float] = (0.26862954, 0.26130258, 0.27577711),
        keep_aspect: bool = True
    ):
        """
        Initialize preprocessor.
        
        Args:
            image_size: Target image size (width, height)
            mean: Normalization mean (ImageNet default)
            std: Normalization std (ImageNet default)
            keep_aspect: Maintain aspect ratio when resizing
        """
        self.image_size = image_size
        self.mean = mean
        self.std = std
        self.keep_aspect = keep_aspect
    
    def __call__(
        self,
        image: Union[Image.Image, str, Path],
        **kwargs
    ) -> ProcessedImage:
        """
        Process image for VLM input.
        
        Args:
            image: Image to process
            **kwargs: Additional options
            
        Returns:
            ProcessedImage with tensor
        """
        # Load image
        img = self.load_image(image)
        original_size = img.size
        
        # Resize
        img = self.resize_image(img, self.image_size, self.keep_aspect)
        
        # Convert to tensor
        img_array = np.array(img).astype(np.float32) / 255.0
        
        # Normalize
        img_array = (img_array - self.mean) / self.std
        
        # Transpose to (C, H, W)
        img_tensor = torch.from_numpy(img_array).permute(2, 0, 1)
        
        # Add batch dimension
        img_tensor = img_tensor.unsqueeze(0)
        
        return ProcessedImage(
            pixel_values=img_tensor,
            image_size=self.image_size,
            original_size=original_size,
            metadata={}
        )
    
    def process_batch(
        self,
        images: List[Union[Image.Image, str, Path]]
    ) -> ProcessedImage:
        """
        Process a batch of images.
        
        Args:
            images: List of images
            
        Returns:
            ProcessedImage with batched tensor
        """
        processed = [self(img) for img in images]
        
        # Stack tensors
        pixel_values = torch.cat([p.pixel_values for p in processed], dim=0)
        
        return ProcessedImage(
            pixel_values=pixel_values,
            image_size=self.image_size,
            original_size=processed[0].original_size,
            metadata={'batch_size': len(images)}
        )


class QwenVLProcessor(ImageProcessor):
    """
    Image processor for Qwen-VL models.
    
    Uses the official Qwen-VL preprocessing pipeline.
    """
    
    def __init__(self, model_name: str = "Qwen/Qwen2-VL-7B-Instruct"):
        """
        Initialize Qwen-VL processor.
        
        Args:
            model_name: Qwen-VL model name
        """
        self.model_name = model_name
        self._processor = None
    
    def _load_processor(self):
        """Lazy load the processor."""
        if self._processor is not None:
            return
        
        try:
            from transformers import AutoProcessor
            self._processor = AutoProcessor.from_pretrained(
                self.model_name,
                trust_remote_code=True
            )
        except Exception as e:
            print(f"Warning: Could not load Qwen-VL processor: {e}")
            print("Falling back to generic processor")
            self._processor = VLMPreprocessor()
    
    def __call__(
        self,
        image: Union[Image.Image, str, Path],
        text: Optional[str] = None,
        **kwargs
    ) -> ProcessedImage:
        """
        Process image for Qwen-VL.
        
        Args:
            image: Image to process
            text: Optional text prompt
            **kwargs: Additional options
            
        Returns:
            ProcessedImage
        """
        self._load_processor()
        
        # Load image
        img = self.load_image(image)
        original_size = img.size
        
        if isinstance(self._processor, VLMPreprocessor):
            return self._processor(img)
        
        # Use official processor
        if text:
            inputs = self._processor(text=text, images=img, return_tensors="pt")
        else:
            inputs = self._processor(images=img, return_tensors="pt")
        
        return ProcessedImage(
            pixel_values=inputs.get('pixel_values', inputs.get('image')),
            image_size=img.size,
            original_size=original_size,
            metadata={'input_ids': inputs.get('input_ids')}
        )


class LLaVAProcessor(ImageProcessor):
    """
    Image processor for LLaVA models.
    
    Uses the official LLaVA preprocessing pipeline.
    """
    
    def __init__(self, model_name: str = "llava-hf/llava-v1.6-34b-hf"):
        """
        Initialize LLaVA processor.
        
        Args:
            model_name: LLaVA model name
        """
        self.model_name = model_name
        self._processor = None
    
    def _load_processor(self):
        """Lazy load the processor."""
        if self._processor is not None:
            return
        
        try:
            from transformers import AutoProcessor
            self._processor = AutoProcessor.from_pretrained(
                self.model_name,
                trust_remote_code=True
            )
        except Exception as e:
            print(f"Warning: Could not load LLaVA processor: {e}")
            print("Falling back to generic processor")
            self._processor = VLMPreprocessor(image_size=(336, 336))
    
    def __call__(
        self,
        image: Union[Image.Image, str, Path],
        text: Optional[str] = None,
        **kwargs
    ) -> ProcessedImage:
        """
        Process image for LLaVA.
        
        Args:
            image: Image to process
            text: Optional text prompt
            **kwargs: Additional options
            
        Returns:
            ProcessedImage
        """
        self._load_processor()
        
        # Load image
        img = self.load_image(image)
        original_size = img.size
        
        if isinstance(self._processor, VLMPreprocessor):
            return self._processor(img)
        
        # Use official processor
        if text:
            inputs = self._processor(text=text, images=img, return_tensors="pt")
        else:
            inputs = self._processor(images=img, return_tensors="pt")
        
        return ProcessedImage(
            pixel_values=inputs.get('pixel_values'),
            image_size=img.size,
            original_size=original_size,
            metadata={'input_ids': inputs.get('input_ids')}
        )


# Processor registry
VLM_PROCESSORS = {
    'generic': VLMPreprocessor,
    'qwen_vl': QwenVLProcessor,
    'llava': LLaVAProcessor,
}


def get_processor(
    model_type: str = "generic",
    **kwargs
) -> ImageProcessor:
    """
    Get image processor for VLM model type.
    
    Args:
        model_type: Model type (generic, qwen_vl, llava)
        **kwargs: Processor arguments
        
    Returns:
        ImageProcessor instance
    """
    if model_type not in VLM_PROCESSORS:
        print(f"Unknown processor type: {model_type}, using generic")
        model_type = "generic"
    
    return VLM_PROCESSORS[model_type](**kwargs)


def detect_processor_type(model_name: str) -> str:
    """
    Detect appropriate processor type from model name.
    
    Args:
        model_name: Model name or path
        
    Returns:
        Processor type string
    """
    model_lower = model_name.lower()
    
    if 'qwen' in model_lower and ('vl' in model_lower or 'vision' in model_lower):
        return 'qwen_vl'
    elif 'llava' in model_lower:
        return 'llava'
    else:
        return 'generic'
