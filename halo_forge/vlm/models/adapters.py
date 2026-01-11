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
        from transformers import AutoTokenizer, AutoProcessor
        
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
        
        # Load model - use appropriate class for Qwen2-VL
        device_map = "auto" if self.device == "auto" else self.device
        
        # Qwen2-VL requires Qwen2VLForConditionalGeneration, not AutoModelForCausalLM
        if "Qwen2-VL" in self.model_name or "qwen2-vl" in self.model_name.lower():
            from transformers import Qwen2VLForConditionalGeneration
            self.model = Qwen2VLForConditionalGeneration.from_pretrained(
                self.model_name,
                torch_dtype=self.dtype,
                device_map=device_map,
                trust_remote_code=self.trust_remote_code
            )
        else:
            # Fallback for older Qwen-VL models
            from transformers import AutoModelForCausalLM
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
            print("Trying CausalLM with separate component loading...")
            
            from transformers import AutoModelForCausalLM, AutoTokenizer, AutoImageProcessor
            
            # Try loading processor - if it fails, load components separately
            try:
                self.processor = AutoProcessor.from_pretrained(
                    self.model_name,
                    trust_remote_code=self.trust_remote_code
                )
            except ValueError as proc_err:
                # Processor failed (e.g., custom tokenizer class not found)
                # Try loading image processor directly
                print(f"AutoProcessor failed: {proc_err}")
                print("Loading image processor and tokenizer separately...")
                
                try:
                    self.processor = AutoImageProcessor.from_pretrained(
                        self.model_name,
                        trust_remote_code=self.trust_remote_code
                    )
                except Exception:
                    # Fallback: Create a minimal processor wrapper
                    self.processor = None
            
            # Load tokenizer - try with and without trust_remote_code
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(
                    self.model_name,
                    trust_remote_code=self.trust_remote_code
                )
            except ValueError:
                # Try loading with use_fast=False or from base model
                print("Trying slow tokenizer...")
                try:
                    self.tokenizer = AutoTokenizer.from_pretrained(
                        self.model_name,
                        trust_remote_code=self.trust_remote_code,
                        use_fast=False
                    )
                except Exception as tok_err:
                    raise ValueError(
                        f"Could not load tokenizer for {self.model_name}. "
                        f"This model may require a specific transformers version or "
                        f"custom dependencies. Error: {tok_err}"
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
        
        # Process inputs - handle different processor types
        if self.processor is not None and hasattr(self.processor, '__call__'):
            # Check if processor supports both text and images
            try:
                inputs = self.processor(
                    text=prompt,
                    images=img,
                    return_tensors="pt"
                )
            except TypeError:
                # Processor might be image-only, process separately
                image_inputs = self.processor(images=img, return_tensors="pt")
                text_inputs = self.tokenizer(prompt, return_tensors="pt")
                inputs = {**image_inputs, **text_inputs}
        else:
            # No processor - use tokenizer and manual image handling
            text_inputs = self.tokenizer(prompt, return_tensors="pt")
            # For models without proper processor, embed image info in prompt
            inputs = text_inputs
        
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


class LFMVLAdapter(VLMAdapter):
    """
    Adapter for LiquidAI LFM VL models.
    
    LFM models use a custom tokenizer backend that requires special handling.
    """
    
    def load(self):
        """Load LFM VL model with custom tokenizer handling."""
        from transformers import (
            AutoModelForCausalLM, 
            AutoTokenizer, 
            AutoImageProcessor,
            CLIPImageProcessor
        )
        
        print(f"Loading LFM VL model: {self.model_name}")
        
        # LFM models have a custom TokenizersBackend that may not be available
        # Try loading tokenizer with different strategies
        tokenizer_loaded = False
        
        # Strategy 1: Try with trust_remote_code and use_fast=False
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                trust_remote_code=True,
                use_fast=False
            )
            tokenizer_loaded = True
        except (ValueError, OSError) as e:
            print(f"Slow tokenizer loading failed: {e}")
        
        # Strategy 2: Try to find a base tokenizer in the model config
        if not tokenizer_loaded:
            try:
                # Many VL models use a LLaMA or similar base tokenizer
                from transformers import AutoConfig
                config = AutoConfig.from_pretrained(
                    self.model_name, 
                    trust_remote_code=True
                )
                
                # Check if there's a text_config with tokenizer info
                if hasattr(config, 'text_config'):
                    base_model = getattr(config.text_config, '_name_or_path', None)
                    if base_model:
                        print(f"Trying base tokenizer from: {base_model}")
                        self.tokenizer = AutoTokenizer.from_pretrained(base_model)
                        tokenizer_loaded = True
            except Exception as e:
                print(f"Config-based tokenizer loading failed: {e}")
        
        # Strategy 3: Use a known compatible tokenizer for LFM models
        if not tokenizer_loaded:
            print("Falling back to LLaMA tokenizer as base...")
            try:
                # LFM models are often based on LLaMA architecture
                self.tokenizer = AutoTokenizer.from_pretrained(
                    "meta-llama/Llama-2-7b-hf",
                    trust_remote_code=True
                )
                tokenizer_loaded = True
            except Exception as e:
                print(f"LLaMA tokenizer fallback failed: {e}")
        
        if not tokenizer_loaded:
            raise ValueError(
                f"Could not load tokenizer for {self.model_name}. "
                f"LFM VL models may require a specific transformers version or "
                f"the 'tokenizers' package to be updated. "
                f"Try: pip install --upgrade tokenizers transformers"
            )
        
        # Load image processor
        try:
            self.processor = AutoImageProcessor.from_pretrained(
                self.model_name,
                trust_remote_code=True
            )
        except Exception:
            print("AutoImageProcessor failed, using CLIP processor...")
            self.processor = CLIPImageProcessor.from_pretrained(
                "openai/clip-vit-base-patch32"
            )
        
        # Load model
        device_map = "auto" if self.device == "auto" else self.device
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=self.dtype,
            device_map=device_map,
            trust_remote_code=True
        )
        
        self._loaded = True
        print(f"Loaded LFM VL on {self.model.device}")
    
    def generate(
        self,
        image: Union[Image.Image, str, Path],
        prompt: str,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        do_sample: bool = True,
        **kwargs
    ) -> VLMOutput:
        """Generate with LFM VL model."""
        self._ensure_loaded()
        
        # Load image
        img = self._load_image(image)
        
        # Process image
        image_inputs = self.processor(images=img, return_tensors="pt")
        
        # Tokenize text
        text_inputs = self.tokenizer(prompt, return_tensors="pt")
        
        # Combine inputs
        inputs = {**image_inputs, **text_inputs}
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
        generated_text = self.tokenizer.decode(
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
    'lfm_vl': LFMVLAdapter,
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
    elif 'lfm' in model_lower or 'liquidai' in model_lower:
        return 'lfm_vl'
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
        'lfm_vl': [
            'LiquidAI/LFM2.5-VL-1.6B',
            'LiquidAI/LFM2.5-VL-3.2B',
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
