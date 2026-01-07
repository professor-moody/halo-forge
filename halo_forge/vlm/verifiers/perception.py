"""
Perception Checker

Verifies visual perception accuracy in VLM outputs.
Uses object detection (YOLOv8) and OCR (EasyOCR) to validate claims.
"""

import re
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple, Set
from pathlib import Path

from PIL import Image
import numpy as np

from halo_forge.rlvr.verifiers.base import VerifyResult


@dataclass
class Detection:
    """Object detection result."""
    label: str
    confidence: float
    bbox: Tuple[float, float, float, float]  # x1, y1, x2, y2
    center: Tuple[float, float]  # center x, y


@dataclass
class PerceptionResult:
    """Result of perception verification."""
    object_score: float
    text_score: float
    spatial_score: float
    counting_score: float
    overall_score: float
    details: Dict[str, Any]


class PerceptionChecker:
    """
    Verifies visual perception claims in VLM outputs.
    
    Verification Types:
    1. Object Detection - Are claimed objects in the image?
    2. OCR - Is extracted text accurate?
    3. Spatial Reasoning - Are spatial relationships correct?
    4. Counting - Are object counts accurate?
    
    Usage:
        checker = PerceptionChecker()
        result = checker.verify(image, completion)
    """
    
    # Common object synonyms for matching
    SYNONYMS = {
        'car': {'car', 'automobile', 'vehicle', 'sedan'},
        'person': {'person', 'man', 'woman', 'human', 'people', 'pedestrian'},
        'dog': {'dog', 'puppy', 'canine'},
        'cat': {'cat', 'kitten', 'feline'},
        'phone': {'phone', 'cellphone', 'mobile', 'smartphone'},
        'laptop': {'laptop', 'computer', 'notebook'},
        'book': {'book', 'textbook', 'novel'},
        'cup': {'cup', 'mug', 'glass'},
        'bottle': {'bottle', 'container'},
        'chair': {'chair', 'seat'},
        'table': {'table', 'desk'},
    }
    
    # Spatial relationship patterns
    SPATIAL_PATTERNS = {
        'left': r'(?:to the )?left of|on the left',
        'right': r'(?:to the )?right of|on the right',
        'above': r'above|over|on top of',
        'below': r'below|under|beneath',
        'next_to': r'next to|beside|adjacent to',
        'in_front': r'in front of|before',
        'behind': r'behind|in back of',
    }
    
    def __init__(
        self,
        detector_model: str = "yolov8n",
        confidence_threshold: float = 0.25,
        use_ocr: bool = True,
        max_workers: int = 1
    ):
        """
        Initialize perception checker.
        
        Args:
            detector_model: YOLOv8 model variant (yolov8n, yolov8s, yolov8m)
            confidence_threshold: Minimum detection confidence
            use_ocr: Whether to use OCR for text verification
            max_workers: Parallel workers (1 recommended for GPU)
        """
        self.detector_model = detector_model
        self.confidence_threshold = confidence_threshold
        self.use_ocr = use_ocr
        self.max_workers = max_workers
        
        self._detector = None
        self._ocr = None
        self._detector_loaded = False
        self._ocr_loaded = False
    
    def _load_detector(self):
        """Lazy load YOLOv8 detector."""
        if self._detector_loaded:
            return
        
        try:
            from ultralytics import YOLO
            self._detector = YOLO(f"{self.detector_model}.pt")
            self._detector_loaded = True
        except ImportError:
            print("Warning: ultralytics not installed. Object detection disabled.")
            print("Install with: pip install ultralytics")
            self._detector = None
            self._detector_loaded = True
    
    def _load_ocr(self):
        """Lazy load EasyOCR."""
        if self._ocr_loaded:
            return
        
        if not self.use_ocr:
            self._ocr_loaded = True
            return
        
        try:
            import easyocr
            self._ocr = easyocr.Reader(['en'], gpu=True)
            self._ocr_loaded = True
        except ImportError:
            print("Warning: easyocr not installed. OCR verification disabled.")
            print("Install with: pip install easyocr")
            self._ocr = None
            self._ocr_loaded = True
    
    def detect_objects(self, image: Image.Image) -> List[Detection]:
        """
        Run object detection on image.
        
        Args:
            image: PIL Image
            
        Returns:
            List of Detection objects
        """
        self._load_detector()
        
        if self._detector is None:
            return []
        
        # Convert to numpy if needed
        if isinstance(image, Image.Image):
            image_np = np.array(image)
        else:
            image_np = image
        
        # Run detection
        results = self._detector(image_np, verbose=False)
        
        detections = []
        for result in results:
            boxes = result.boxes
            for box in boxes:
                conf = float(box.conf[0])
                if conf < self.confidence_threshold:
                    continue
                
                cls_id = int(box.cls[0])
                label = result.names[cls_id]
                
                # Get bbox coordinates
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                center_x = (x1 + x2) / 2
                center_y = (y1 + y2) / 2
                
                detections.append(Detection(
                    label=label.lower(),
                    confidence=conf,
                    bbox=(x1, y1, x2, y2),
                    center=(center_x, center_y)
                ))
        
        return detections
    
    def extract_text(self, image: Image.Image) -> List[str]:
        """
        Extract text from image using OCR.
        
        Args:
            image: PIL Image
            
        Returns:
            List of detected text strings
        """
        self._load_ocr()
        
        if self._ocr is None:
            return []
        
        # Convert to numpy
        if isinstance(image, Image.Image):
            image_np = np.array(image)
        else:
            image_np = image
        
        # Run OCR
        results = self._ocr.readtext(image_np)
        
        # Extract text strings
        texts = [text for (_, text, _) in results]
        
        return texts
    
    def extract_object_claims(self, completion: str) -> List[str]:
        """
        Extract object claims from completion text.
        
        Looks for phrases like "I see a cat", "there is a dog", etc.
        
        Args:
            completion: Model completion text
            
        Returns:
            List of claimed objects
        """
        claims = []
        
        # Common claim patterns
        patterns = [
            r"(?:I (?:can )?see|there (?:is|are)|shows?|contains?|has|features?)\s+(?:a|an|the|some)?\s*(\w+)",
            r"(?:image|picture|photo) (?:of|shows?|contains?|depicts?)\s+(?:a|an|the)?\s*(\w+)",
            r"(?:a|an)\s+(\w+)\s+(?:is|are)\s+(?:visible|present|shown)",
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, completion.lower())
            claims.extend(matches)
        
        # Also look for common objects directly mentioned
        common_objects = [
            'person', 'car', 'dog', 'cat', 'bird', 'table', 'chair', 
            'phone', 'laptop', 'book', 'cup', 'bottle', 'tree', 'building',
            'bicycle', 'motorcycle', 'bus', 'truck', 'train', 'airplane'
        ]
        
        for obj in common_objects:
            if obj in completion.lower():
                claims.append(obj)
        
        return list(set(claims))
    
    def extract_text_claims(self, completion: str) -> List[str]:
        """
        Extract text claims from completion.
        
        Looks for quoted text that claims to be read from the image.
        
        Args:
            completion: Model completion text
            
        Returns:
            List of claimed text strings
        """
        claims = []
        
        # Look for quoted strings
        quoted = re.findall(r'"([^"]+)"', completion)
        claims.extend(quoted)
        
        # Look for "says/reads X" patterns
        says_patterns = re.findall(
            r"(?:says?|reads?|shows?|displays?|written|text)\s*[:\"]?\s*([A-Z][A-Za-z0-9\s]+)",
            completion
        )
        claims.extend(says_patterns)
        
        return list(set(claims))
    
    def extract_counting_claims(self, completion: str) -> Dict[str, int]:
        """
        Extract counting claims from completion.
        
        Looks for phrases like "three cats", "5 people", etc.
        
        Args:
            completion: Model completion text
            
        Returns:
            Dict mapping objects to claimed counts
        """
        counts = {}
        
        # Number words
        number_words = {
            'one': 1, 'two': 2, 'three': 3, 'four': 4, 'five': 5,
            'six': 6, 'seven': 7, 'eight': 8, 'nine': 9, 'ten': 10
        }
        
        # Pattern: "three cats", "5 people"
        patterns = [
            r"(\d+|one|two|three|four|five|six|seven|eight|nine|ten)\s+(\w+s?)",
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, completion.lower())
            for num_str, obj in matches:
                if num_str.isdigit():
                    count = int(num_str)
                else:
                    count = number_words.get(num_str, 1)
                
                # Normalize object name (remove plural)
                obj = obj.rstrip('s') if obj.endswith('s') and len(obj) > 3 else obj
                counts[obj] = count
        
        return counts
    
    def extract_spatial_claims(self, completion: str) -> List[Tuple[str, str, str]]:
        """
        Extract spatial relationship claims.
        
        Looks for phrases like "the dog is left of the cat".
        
        Args:
            completion: Model completion text
            
        Returns:
            List of (object1, relationship, object2) tuples
        """
        claims = []
        
        for rel_name, pattern in self.SPATIAL_PATTERNS.items():
            # Pattern: "X is/are left of Y"
            full_pattern = rf"(?:the\s+)?(\w+)\s+(?:is|are)\s+(?:{pattern})\s+(?:the\s+)?(\w+)"
            matches = re.findall(full_pattern, completion.lower())
            for obj1, obj2 in matches:
                claims.append((obj1, rel_name, obj2))
        
        return claims
    
    def verify_object_claims(
        self,
        claims: List[str],
        detections: List[Detection]
    ) -> Tuple[float, Dict[str, bool]]:
        """
        Verify object claims against detections.
        
        Args:
            claims: List of claimed objects
            detections: List of detections from image
            
        Returns:
            (score, details) where score is 0-1 and details shows each claim result
        """
        if not claims:
            return 1.0, {}
        
        detected_labels = {d.label for d in detections}
        
        # Expand with synonyms
        expanded_detected = set()
        for label in detected_labels:
            expanded_detected.add(label)
            for key, synonyms in self.SYNONYMS.items():
                if label in synonyms:
                    expanded_detected.update(synonyms)
        
        verified = {}
        for claim in claims:
            claim_lower = claim.lower()
            # Check if claim matches any detected object
            matched = claim_lower in expanded_detected
            if not matched:
                # Check synonyms for claim
                for key, synonyms in self.SYNONYMS.items():
                    if claim_lower in synonyms:
                        if any(s in expanded_detected for s in synonyms):
                            matched = True
                            break
            
            verified[claim] = matched
        
        score = sum(verified.values()) / len(verified)
        return score, verified
    
    def verify_text_claims(
        self,
        claims: List[str],
        ocr_texts: List[str]
    ) -> Tuple[float, Dict[str, bool]]:
        """
        Verify text claims against OCR results.
        
        Args:
            claims: List of claimed text strings
            ocr_texts: List of OCR-extracted texts
            
        Returns:
            (score, details)
        """
        if not claims:
            return 1.0, {}
        
        # Combine all OCR text
        all_text = ' '.join(ocr_texts).lower()
        
        verified = {}
        for claim in claims:
            claim_lower = claim.lower()
            # Check if claim appears in OCR text
            verified[claim] = claim_lower in all_text
        
        score = sum(verified.values()) / len(verified)
        return score, verified
    
    def verify_counting_claims(
        self,
        claims: Dict[str, int],
        detections: List[Detection]
    ) -> Tuple[float, Dict[str, Dict]]:
        """
        Verify counting claims against detections.
        
        Args:
            claims: Dict mapping objects to claimed counts
            detections: List of detections
            
        Returns:
            (score, details)
        """
        if not claims:
            return 1.0, {}
        
        # Count detections by label
        detection_counts = {}
        for d in detections:
            label = d.label
            detection_counts[label] = detection_counts.get(label, 0) + 1
        
        verified = {}
        for obj, claimed_count in claims.items():
            actual_count = detection_counts.get(obj, 0)
            
            # Check synonyms
            if actual_count == 0:
                for key, synonyms in self.SYNONYMS.items():
                    if obj in synonyms or key == obj:
                        for syn in synonyms:
                            actual_count = max(actual_count, detection_counts.get(syn, 0))
            
            verified[obj] = {
                'claimed': claimed_count,
                'actual': actual_count,
                'correct': claimed_count == actual_count
            }
        
        score = sum(1 for v in verified.values() if v['correct']) / len(verified)
        return score, verified
    
    def verify_spatial_claims(
        self,
        claims: List[Tuple[str, str, str]],
        detections: List[Detection]
    ) -> Tuple[float, Dict[str, bool]]:
        """
        Verify spatial relationship claims.
        
        Args:
            claims: List of (obj1, relationship, obj2) tuples
            detections: List of detections
            
        Returns:
            (score, details)
        """
        if not claims:
            return 1.0, {}
        
        # Build object -> detection map
        obj_detections = {}
        for d in detections:
            if d.label not in obj_detections:
                obj_detections[d.label] = []
            obj_detections[d.label].append(d)
        
        verified = {}
        for obj1, rel, obj2 in claims:
            claim_key = f"{obj1} {rel} {obj2}"
            
            # Find detections for both objects
            d1_list = obj_detections.get(obj1, [])
            d2_list = obj_detections.get(obj2, [])
            
            if not d1_list or not d2_list:
                verified[claim_key] = False
                continue
            
            # Check if relationship holds for any pair
            valid = False
            for d1 in d1_list:
                for d2 in d2_list:
                    if self._check_spatial_relation(d1, d2, rel):
                        valid = True
                        break
                if valid:
                    break
            
            verified[claim_key] = valid
        
        score = sum(verified.values()) / len(verified)
        return score, verified
    
    def _check_spatial_relation(
        self,
        d1: Detection,
        d2: Detection,
        relation: str
    ) -> bool:
        """Check if d1 has the specified spatial relation to d2."""
        c1 = d1.center
        c2 = d2.center
        
        if relation == 'left':
            return c1[0] < c2[0]
        elif relation == 'right':
            return c1[0] > c2[0]
        elif relation == 'above':
            return c1[1] < c2[1]
        elif relation == 'below':
            return c1[1] > c2[1]
        elif relation == 'next_to':
            # Within 100 pixels horizontally
            return abs(c1[0] - c2[0]) < 100
        elif relation == 'in_front':
            return c1[1] > c2[1]  # In images, "in front" often means lower
        elif relation == 'behind':
            return c1[1] < c2[1]
        
        return False
    
    def verify(
        self,
        image: Image.Image,
        completion: str
    ) -> PerceptionResult:
        """
        Verify all perception claims in a VLM completion.
        
        Args:
            image: Input image
            completion: Model completion text
            
        Returns:
            PerceptionResult with scores for each verification type
        """
        # Run detection and OCR
        detections = self.detect_objects(image)
        ocr_texts = self.extract_text(image) if self.use_ocr else []
        
        # Extract claims
        object_claims = self.extract_object_claims(completion)
        text_claims = self.extract_text_claims(completion)
        counting_claims = self.extract_counting_claims(completion)
        spatial_claims = self.extract_spatial_claims(completion)
        
        # Verify each type
        object_score, object_details = self.verify_object_claims(object_claims, detections)
        text_score, text_details = self.verify_text_claims(text_claims, ocr_texts)
        counting_score, counting_details = self.verify_counting_claims(counting_claims, detections)
        spatial_score, spatial_details = self.verify_spatial_claims(spatial_claims, detections)
        
        # Calculate overall score (weighted average)
        weights = {
            'object': 0.4,
            'text': 0.2,
            'spatial': 0.2,
            'counting': 0.2,
        }
        
        # Adjust weights based on what claims exist
        active_weights = {}
        if object_claims:
            active_weights['object'] = weights['object']
        if text_claims:
            active_weights['text'] = weights['text']
        if spatial_claims:
            active_weights['spatial'] = weights['spatial']
        if counting_claims:
            active_weights['counting'] = weights['counting']
        
        if active_weights:
            total_weight = sum(active_weights.values())
            normalized_weights = {k: v / total_weight for k, v in active_weights.items()}
            
            overall = 0.0
            if 'object' in normalized_weights:
                overall += object_score * normalized_weights['object']
            if 'text' in normalized_weights:
                overall += text_score * normalized_weights['text']
            if 'spatial' in normalized_weights:
                overall += spatial_score * normalized_weights['spatial']
            if 'counting' in normalized_weights:
                overall += counting_score * normalized_weights['counting']
        else:
            # No claims to verify, give full score
            overall = 1.0
        
        return PerceptionResult(
            object_score=object_score,
            text_score=text_score,
            spatial_score=spatial_score,
            counting_score=counting_score,
            overall_score=overall,
            details={
                'detections': [d.label for d in detections],
                'ocr_texts': ocr_texts,
                'object_claims': object_details,
                'text_claims': text_details,
                'counting_claims': counting_details,
                'spatial_claims': spatial_details,
            }
        )
