"""
Vision Verifier

Multi-stage verification for vision-language model outputs.
Combines perception, reasoning, and output verification.
"""

import warnings
from dataclasses import dataclass
from typing import Dict, Any, Optional, List, Union
from pathlib import Path

from PIL import Image

from halo_forge.rlvr.verifiers.base import Verifier, VerifyResult
from halo_forge.vlm.verifiers.perception import PerceptionChecker, PerceptionResult
from halo_forge.vlm.verifiers.reasoning import ReasoningChecker, ReasoningResult
from halo_forge.vlm.verifiers.output import OutputChecker, OutputResult


class VLMVerificationError(Exception):
    """Base exception for VLM verification errors."""
    pass


class ImageLoadError(VLMVerificationError):
    """Raised when image loading fails."""
    pass


class DependencyWarning(UserWarning):
    """Warning for missing optional dependencies."""
    pass


def check_vlm_dependencies() -> Dict[str, bool]:
    """
    Check which VLM dependencies are available.
    
    Returns:
        Dictionary mapping dependency name to availability status
    """
    deps = {
        "ultralytics": False,  # YOLOv8
        "easyocr": False,
        "pillow": True,  # Always available (required)
    }
    
    try:
        from ultralytics import YOLO
        deps["ultralytics"] = True
    except ImportError:
        pass
    
    try:
        import easyocr
        deps["easyocr"] = True
    except ImportError:
        pass
    
    return deps


@dataclass
class VisionVerifyResult:
    """Complete verification result for VLM outputs."""
    perception: PerceptionResult
    reasoning: ReasoningResult
    output: Optional[OutputResult]
    overall_reward: float
    success: bool
    details: Dict[str, Any]


class VisionVerifier(Verifier):
    """
    Multi-stage verification for vision-language model outputs.
    
    Verification Pipeline:
    1. Perception - Verify visual claims (objects, text, spatial)
    2. Reasoning - Verify reasoning chain quality
    3. Output - Verify final answer accuracy (if ground truth provided)
    
    Reward Calculation:
    - perception_weight × perception_score
    - reasoning_weight × reasoning_score
    - output_weight × output_score
    
    Usage:
        verifier = VisionVerifier()
        result = verifier.verify(image, prompt, completion, ground_truth="yes")
    """
    
    def __init__(
        self,
        perception_weight: float = 0.3,
        reasoning_weight: float = 0.4,
        output_weight: float = 0.3,
        perception_config: Optional[Dict[str, Any]] = None,
        reasoning_config: Optional[Dict[str, Any]] = None,
        output_config: Optional[Dict[str, Any]] = None,
        max_workers: int = 1
    ):
        """
        Initialize vision verifier.
        
        Args:
            perception_weight: Weight for perception score (0-1)
            reasoning_weight: Weight for reasoning score (0-1)
            output_weight: Weight for output score (0-1)
            perception_config: Config for PerceptionChecker
            reasoning_config: Config for ReasoningChecker
            output_config: Config for OutputChecker
            max_workers: Parallel workers
        """
        super().__init__(max_workers=max_workers)
        
        # Weights (normalize to sum to 1)
        total = perception_weight + reasoning_weight + output_weight
        self.perception_weight = perception_weight / total
        self.reasoning_weight = reasoning_weight / total
        self.output_weight = output_weight / total
        
        # Initialize checkers
        perception_config = perception_config or {}
        reasoning_config = reasoning_config or {}
        output_config = output_config or {}
        
        self.perception_checker = PerceptionChecker(**perception_config)
        self.reasoning_checker = ReasoningChecker(**reasoning_config)
        self.output_checker = OutputChecker(**output_config)
    
    def verify(
        self,
        image: Union[Image.Image, str, Path],
        prompt: str,
        completion: str,
        ground_truth: Optional[str] = None,
        expected_format: Optional[str] = None
    ) -> VerifyResult:
        """
        Verify a VLM completion.
        
        Args:
            image: Input image (PIL Image, path, or URL)
            prompt: Text prompt/question
            completion: Model completion
            ground_truth: Optional expected answer
            expected_format: Optional expected answer format
            
        Returns:
            VerifyResult with combined reward
            
        Raises:
            ImageLoadError: If image cannot be loaded
            VLMVerificationError: If verification fails critically
        """
        # Validate inputs
        if not prompt:
            warnings.warn("Empty prompt provided", DependencyWarning)
        
        if not completion:
            return VerifyResult(
                success=False,
                reward=0.0,
                details="Empty completion provided",
                error="No completion to verify"
            )
        
        # Load image if needed
        try:
            if isinstance(image, (str, Path)):
                image_path = Path(image)
                if str(image).startswith(('http://', 'https://')):
                    import requests
                    from io import BytesIO
                    response = requests.get(str(image), timeout=30)
                    response.raise_for_status()
                    image = Image.open(BytesIO(response.content))
                elif image_path.exists():
                    image = Image.open(image_path)
                else:
                    raise ImageLoadError(f"Image not found: {image}")
        except Exception as e:
            if isinstance(e, ImageLoadError):
                raise
            raise ImageLoadError(f"Failed to load image: {e}") from e
        
        # 1. Perception verification
        perception_result = self.perception_checker.verify(image, completion)
        
        # 2. Reasoning verification
        reasoning_result = self.reasoning_checker.verify_with_context(
            completion, prompt, ground_truth
        )
        
        # 3. Output verification (if ground truth provided)
        if ground_truth:
            output_result = self.output_checker.verify(
                completion, ground_truth, expected_format
            )
        else:
            output_result = None
        
        # Calculate combined reward
        reward = self._calculate_reward(
            perception_result,
            reasoning_result,
            output_result
        )
        
        # Determine success
        success = reward >= 0.5
        
        # Build details
        details = {
            'perception': {
                'score': perception_result.overall_score,
                'object_score': perception_result.object_score,
                'text_score': perception_result.text_score,
                'spatial_score': perception_result.spatial_score,
            },
            'reasoning': {
                'score': reasoning_result.overall_score,
                'structure_score': reasoning_result.structure_score,
                'consistency_score': reasoning_result.consistency_score,
                'grounding_score': reasoning_result.grounding_score,
            },
            'weights': {
                'perception': self.perception_weight,
                'reasoning': self.reasoning_weight,
                'output': self.output_weight,
            }
        }
        
        if output_result:
            details['output'] = {
                'score': output_result.overall_score,
                'exact_match': output_result.exact_match,
                'fuzzy_score': output_result.fuzzy_score,
            }
        
        return VerifyResult(
            success=success,
            reward=reward,
            details=details,
            metadata={
                'perception_result': perception_result,
                'reasoning_result': reasoning_result,
                'output_result': output_result,
            }
        )
    
    def _calculate_reward(
        self,
        perception: PerceptionResult,
        reasoning: ReasoningResult,
        output: Optional[OutputResult]
    ) -> float:
        """
        Calculate combined reward from all verification stages.
        
        Args:
            perception: Perception verification result
            reasoning: Reasoning verification result
            output: Output verification result (may be None)
            
        Returns:
            Combined reward (0-1)
        """
        if output is not None:
            # Full verification with ground truth
            reward = (
                self.perception_weight * perception.overall_score +
                self.reasoning_weight * reasoning.overall_score +
                self.output_weight * output.overall_score
            )
        else:
            # No ground truth - redistribute output weight
            adjusted_perception = self.perception_weight / (self.perception_weight + self.reasoning_weight)
            adjusted_reasoning = self.reasoning_weight / (self.perception_weight + self.reasoning_weight)
            
            reward = (
                adjusted_perception * perception.overall_score +
                adjusted_reasoning * reasoning.overall_score
            )
        
        return min(1.0, max(0.0, reward))
    
    def verify_batch(
        self,
        samples: List[Dict[str, Any]]
    ) -> List[VerifyResult]:
        """
        Verify a batch of VLM samples.
        
        Each sample should have:
        - image: Image or path
        - prompt: Text prompt
        - completion: Model completion
        - ground_truth: Optional expected answer
        
        Args:
            samples: List of sample dicts
            
        Returns:
            List of VerifyResult
        """
        results = []
        
        for sample in samples:
            result = self.verify(
                image=sample['image'],
                prompt=sample['prompt'],
                completion=sample['completion'],
                ground_truth=sample.get('ground_truth'),
                expected_format=sample.get('expected_format')
            )
            results.append(result)
        
        return results
    
    def cleanup(self):
        """Clean up resources."""
        # Perception checker may have loaded models
        if hasattr(self.perception_checker, '_detector'):
            del self.perception_checker._detector
        if hasattr(self.perception_checker, '_ocr'):
            del self.perception_checker._ocr


class VQAVerifier(VisionVerifier):
    """
    Specialized verifier for Visual Question Answering tasks.
    
    Emphasizes output accuracy over reasoning structure.
    """
    
    def __init__(self, **kwargs):
        # Adjust weights for VQA
        kwargs.setdefault('perception_weight', 0.2)
        kwargs.setdefault('reasoning_weight', 0.2)
        kwargs.setdefault('output_weight', 0.6)
        
        super().__init__(**kwargs)


class DocVQAVerifier(VisionVerifier):
    """
    Specialized verifier for Document VQA tasks.
    
    Emphasizes text extraction accuracy.
    """
    
    def __init__(self, **kwargs):
        # Enable OCR, emphasize perception
        perception_config = kwargs.pop('perception_config', {})
        perception_config.setdefault('use_ocr', True)
        
        kwargs.setdefault('perception_weight', 0.4)
        kwargs.setdefault('reasoning_weight', 0.2)
        kwargs.setdefault('output_weight', 0.4)
        kwargs['perception_config'] = perception_config
        
        super().__init__(**kwargs)


class ChartQAVerifier(VisionVerifier):
    """
    Specialized verifier for Chart QA tasks.
    
    Emphasizes numerical accuracy and data extraction.
    """
    
    def __init__(self, **kwargs):
        # Configure for chart understanding
        output_config = kwargs.pop('output_config', {})
        output_config.setdefault('normalize_answers', True)
        
        kwargs.setdefault('perception_weight', 0.3)
        kwargs.setdefault('reasoning_weight', 0.3)
        kwargs.setdefault('output_weight', 0.4)
        kwargs['output_config'] = output_config
        
        super().__init__(**kwargs)
