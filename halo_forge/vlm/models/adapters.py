"""
VLM Model Adapters

Unified interface for different VLM architectures.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Any, Optional, List, Union, Tuple
from pathlib import Path

import torch
from PIL import Image


@dataclass
class VLMOutput:
    """Output from VLM generation."""
    text: str
    logits: Optional[torch.Tensor] = None
    attention: Optional[torch.Tensor] = None
    metadata: Optional[Dict[str, Any]] = None


class VLMAdapter(ABC):
    """
    Abstract base class for VLM model adapters.
    
    Provides unified interface for different VLM architectures.
    """
    
    def __init__(
        self,
        model_name: str,
        device: str = "auto",
        dtype: torch.dtype = torch.bfloat16,
        trust_remote_code: bool = True
    ):
        """
        Initialize adapter.
        
        Args:
            model_name: Model name or path
            device: Device to load on
            dtype: Model dtype
            trust_remote_code: Trust remote code
        """
        self.model_name = model_name
        self.device = device
        self.dtype = dtype
        self.trust_remote_code = trust_remote_code
        
        self.model = None
        self.tokenizer = None
        self.processor = None
        self._loaded = False
    
    @abstractmethod
    def load(self):
        """Load model, tokenizer, and processor."""
        pass
    
    @abstractmethod
    def generate(
        self,
        image: Union[Image.Image, str, Path],
        prompt: str,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        do_sample: bool = True,
        **kwargs
    ) -> VLMOutput:
        """
        Generate response for image + prompt.
        
        Args:
            image: Input image
            prompt: Text prompt
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            do_sample: Whether to use sampling
            **kwargs: Additional generation args
            
        Returns:
            VLMOutput with generated text
        """
        pass
    
    def generate_batch(
        self,
        images: List[Union[Image.Image, str, Path]],
        prompts: List[str],
        **kwargs
    ) -> List[VLMOutput]:
        """
        Generate responses for batch of images + prompts.
        
        Args:
            images: List of images
            prompts: List of prompts
            **kwargs: Generation args
            
        Returns:
            List of VLMOutput
        """
        results = []
        for image, prompt in zip(images, prompts):
            result = self.generate(image, prompt, **kwargs)
            results.append(result)
        return results
    
    def _ensure_loaded(self):
        """Ensure model is loaded."""
        if not self._loaded:
            self.load()
    
    def _load_image(self, image: Union[Image.Image, str, Path]) -> Image.Image:
        """Load image from path if needed."""
        if isinstance(image, Image.Image):
            return image
        return Image.open(image).convert('RGB')
    
    def cleanup(self):
        """Clean up resources."""
        if self.model is not None:
            del self.model
            self.model = None
        if self.tokenizer is not None:
            del self.tokenizer
            self.tokenizer = None
        if self.processor is not None:
            del self.processor
            self.processor = None
        
        self._loaded = False
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


class QwenVLAdapter(VLMAdapter):
    """
    Adapter for Qwen-VL and Qwen2-VL models.
    
    Supports:
    - Qwen/Qwen-VL-Chat
    - Qwen/Qwen2-VL-7B-Instruct
    - Qwen/Qwen2-VL-72B-Instruct
    """
    
    def __init__(
        self,
        model_name: str = "Qwen/Qwen2-VL-7B-Instruct",
        **kwargs
    ):
        super().__init__(model_name, **kwargs)
    
    def load(self):
        """Load Qwen-VL model."""
        from transformers import AutoModelForCausalLM, AutoTokenizer, AutoProcessor
        
        print(f"Loading Qwen-VL: {self.model_name}")
        
        # Load processor
        self.processor = AutoProcessor.from_pretrained(
            self.model_name,
            trust_remote_code=self.trust_remote_code
        )
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=self.trust_remote_code
        )
        
        # Load model
        device_map = "auto" if self.device == "auto" else self.device
        
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=self.dtype,
            device_map=device_map,
            trust_remote_code=self.trust_remote_code
        )
        
        self._loaded = True
        print(f"Loaded Qwen-VL on {self.model.device}")
    
    def generate(
        self,
        image: Union[Image.Image, str, Path],
        prompt: str,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        do_sample: bool = True,
        **kwargs
    ) -> VLMOutput:
        """Generate with Qwen-VL."""
        self._ensure_loaded()
        
        # Load image
        img = self._load_image(image)
        
        # Format message for Qwen2-VL
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": img},
                    {"type": "text", "text": prompt}
                ]
            }
        ]
        
        # Apply chat template
        text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        # Process inputs
        inputs = self.processor(
            text=[text],
            images=[img],
            return_tensors="pt",
            padding=True
        )
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature if do_sample else None,
                do_sample=do_sample,
                **kwargs
            )
        
        # Decode
        generated_ids = outputs[0][inputs['input_ids'].shape[1]:]
        generated_text = self.tokenizer.decode(
            generated_ids,
            skip_special_tokens=True
        )
        
        return VLMOutput(
            text=generated_text,
            metadata={'model': self.model_name}
        )


class LLaVAAdapter(VLMAdapter):
    """
    Adapter for LLaVA models.
    
    Supports:
    - llava-hf/llava-1.5-7b-hf
    - llava-hf/llava-v1.6-34b-hf
    - llava-hf/LLaVA-NeXT-Video-7B-hf
    """
    
    def __init__(
        self,
        model_name: str = "llava-hf/llava-1.5-7b-hf",
        **kwargs
    ):
        super().__init__(model_name, **kwargs)
    
    def load(self):
        """Load LLaVA model."""
        from transformers import AutoProcessor, LlavaForConditionalGeneration
        
        print(f"Loading LLaVA: {self.model_name}")
        
        # Load processor
        self.processor = AutoProcessor.from_pretrained(
            self.model_name,
            trust_remote_code=self.trust_remote_code
        )
        
        # Load model
        device_map = "auto" if self.device == "auto" else self.device
        
        self.model = LlavaForConditionalGeneration.from_pretrained(
            self.model_name,
            torch_dtype=self.dtype,
            device_map=device_map,
            trust_remote_code=self.trust_remote_code
        )
        
        self.tokenizer = self.processor.tokenizer
        
        self._loaded = True
        print(f"Loaded LLaVA on {self.model.device}")
    
    def generate(
        self,
        image: Union[Image.Image, str, Path],
        prompt: str,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        do_sample: bool = True,
        **kwargs
    ) -> VLMOutput:
        """Generate with LLaVA."""
        self._ensure_loaded()
        
        # Load image
        img = self._load_image(image)
        
        # Format conversation for LLaVA
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": prompt}
                ]
            }
        ]
        
        # Apply chat template
        text = self.processor.apply_chat_template(
            conversation,
            add_generation_prompt=True
        )
        
        # Process inputs
        inputs = self.processor(
            images=img,
            text=text,
            return_tensors="pt"
        )
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature if do_sample else None,
                do_sample=do_sample,
                **kwargs
            )
        
        # Decode
        generated_ids = outputs[0][inputs['input_ids'].shape[1]:]
        generated_text = self.tokenizer.decode(
            generated_ids,
            skip_special_tokens=True
        )
        
        return VLMOutput(
            text=generated_text,
            metadata={'model': self.model_name}
        )


class GenericVLMAdapter(VLMAdapter):
    """
    Generic adapter that attempts to work with any HuggingFace VLM.
    
    Uses AutoModel and AutoProcessor for compatibility.
    """
    
    def load(self):
        """Load generic VLM."""
        from transformers import AutoModelForVision2Seq, AutoProcessor
        
        print(f"Loading VLM: {self.model_name}")
        
        try:
            # Try Vision2Seq first
            self.processor = AutoProcessor.from_pretrained(
                self.model_name,
                trust_remote_code=self.trust_remote_code
            )
            
            device_map = "auto" if self.device == "auto" else self.device
            
            self.model = AutoModelForVision2Seq.from_pretrained(
                self.model_name,
                torch_dtype=self.dtype,
                device_map=device_map,
                trust_remote_code=self.trust_remote_code
            )
            
            self.tokenizer = getattr(self.processor, 'tokenizer', None)
            
        except Exception as e:
            print(f"Vision2Seq loading failed: {e}")
            print("Trying CausalLM...")
            
            from transformers import AutoModelForCausalLM, AutoTokenizer
            
            self.processor = AutoProcessor.from_pretrained(
                self.model_name,
                trust_remote_code=self.trust_remote_code
            )
            
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                trust_remote_code=self.trust_remote_code
            )
            
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=self.dtype,
                device_map="auto",
                trust_remote_code=self.trust_remote_code
            )
        
        self._loaded = True
        print(f"Loaded VLM on {self.model.device}")
    
    def generate(
        self,
        image: Union[Image.Image, str, Path],
        prompt: str,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        do_sample: bool = True,
        **kwargs
    ) -> VLMOutput:
        """Generate with generic VLM."""
        self._ensure_loaded()
        
        # Load image
        img = self._load_image(image)
        
        # Process inputs
        inputs = self.processor(
            text=prompt,
            images=img,
            return_tensors="pt"
        )
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature if do_sample else None,
                do_sample=do_sample,
                **kwargs
            )
        
        # Decode
        if self.tokenizer:
            generated_text = self.tokenizer.decode(
                outputs[0],
                skip_special_tokens=True
            )
        else:
            generated_text = self.processor.decode(
                outputs[0],
                skip_special_tokens=True
            )
        
        return VLMOutput(
            text=generated_text,
            metadata={'model': self.model_name}
        )


# Adapter registry
VLM_ADAPTERS = {
    'qwen_vl': QwenVLAdapter,
    'llava': LLaVAAdapter,
    'generic': GenericVLMAdapter,
}


def get_vlm_adapter(
    model_name: str,
    adapter_type: Optional[str] = None,
    **kwargs
) -> VLMAdapter:
    """
    Get appropriate VLM adapter for model.
    
    Args:
        model_name: Model name or path
        adapter_type: Force specific adapter type
        **kwargs: Adapter arguments
        
    Returns:
        VLMAdapter instance
    """
    if adapter_type is None:
        # Auto-detect adapter type
        adapter_type = detect_adapter_type(model_name)
    
    if adapter_type not in VLM_ADAPTERS:
        print(f"Unknown adapter type: {adapter_type}, using generic")
        adapter_type = 'generic'
    
    adapter_cls = VLM_ADAPTERS[adapter_type]
    return adapter_cls(model_name=model_name, **kwargs)


def detect_adapter_type(model_name: str) -> str:
    """
    Detect appropriate adapter type from model name.
    
    Args:
        model_name: Model name
        
    Returns:
        Adapter type string
    """
    model_lower = model_name.lower()
    
    if 'qwen' in model_lower and ('vl' in model_lower or 'vision' in model_lower):
        return 'qwen_vl'
    elif 'llava' in model_lower:
        return 'llava'
    else:
        return 'generic'


def list_supported_vlms() -> Dict[str, List[str]]:
    """
    List supported VLM models by adapter type.
    
    Returns:
        Dict mapping adapter types to example models
    """
    return {
        'qwen_vl': [
            'Qwen/Qwen-VL-Chat',
            'Qwen/Qwen2-VL-2B-Instruct',
            'Qwen/Qwen2-VL-7B-Instruct',
            'Qwen/Qwen2-VL-72B-Instruct',
        ],
        'llava': [
            'llava-hf/llava-1.5-7b-hf',
            'llava-hf/llava-1.5-13b-hf',
            'llava-hf/llava-v1.6-34b-hf',
            'llava-hf/LLaVA-NeXT-Video-7B-hf',
        ],
        'generic': [
            'microsoft/Florence-2-large',
            'Salesforce/blip2-flan-t5-xl',
        ]
    }
