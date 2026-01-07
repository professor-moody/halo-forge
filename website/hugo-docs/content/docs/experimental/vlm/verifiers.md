---
title: "VLM Verifiers"
weight: 10
---

# Vision-Language Verifiers

Multi-stage verification for VLM outputs with perception-aware rewards.

## VisionVerifier

The main verifier that combines all stages:

```python
from halo_forge.vlm.verifiers import VisionVerifier

verifier = VisionVerifier(
    perception_weight=0.3,
    reasoning_weight=0.4,
    output_weight=0.3,
)

result = verifier.verify(
    image=image,
    prompt="What text is on the sign?",
    completion="The sign says 'STOP'.",
    ground_truth="STOP"
)

print(f"Reward: {result.reward}")
print(f"Success: {result.success}")
```

## Perception Checker

Validates visual claims using object detection and OCR.

### Object Detection

Uses YOLOv8 to verify objects claimed in the completion:

```python
from halo_forge.vlm.verifiers import PerceptionChecker

checker = PerceptionChecker(
    detector_model="yolov8n",      # yolov8n, yolov8s, yolov8m
    confidence_threshold=0.25,
    use_ocr=True
)

result = checker.verify(image, completion)
print(f"Object score: {result.object_score}")
print(f"Text score: {result.text_score}")
print(f"Spatial score: {result.spatial_score}")
```

### Claim Extraction

The checker extracts claims from completions:

- **Object claims**: "I see a cat", "there is a dog"
- **Text claims**: Quoted text, "says X", "reads Y"
- **Counting claims**: "three cats", "5 people"
- **Spatial claims**: "the dog is left of the cat"

### Verification Process

1. Run object detection on image
2. Run OCR on image
3. Extract claims from completion
4. Match claims against detections
5. Calculate per-category scores

## Reasoning Checker

Validates chain-of-thought quality.

```python
from halo_forge.vlm.verifiers import ReasoningChecker

checker = ReasoningChecker(
    min_steps=2,
    require_evidence=True,
    require_conclusion=True
)

result = checker.verify(completion)
print(f"Structure: {result.structure_score}")
print(f"Consistency: {result.consistency_score}")
print(f"Grounding: {result.grounding_score}")
```

### What It Checks

| Aspect | What It Looks For | Weight |
|--------|-------------------|--------|
| Structure | Numbered steps, logical progression | 0.3 |
| Consistency | No contradictions, logical flow | 0.3 |
| Grounding | References to image evidence | 0.4 |

### Evidence Patterns

The checker looks for phrases that reference visual evidence:

- "I can see..."
- "Looking at the image..."
- "The picture shows..."
- "Based on the image..."

## Output Checker

Validates final answer accuracy.

```python
from halo_forge.vlm.verifiers import OutputChecker

checker = OutputChecker(
    fuzzy_threshold=0.8,
    use_semantic=False,
    normalize_answers=True
)

result = checker.verify(
    completion="The answer is STOP",
    ground_truth="stop"
)

print(f"Exact match: {result.exact_match}")
print(f"Fuzzy score: {result.fuzzy_score}")
```

### Match Types

| Type | Method | Score |
|------|--------|-------|
| Exact | Normalized string comparison | 1.0 |
| Fuzzy | SequenceMatcher ratio | 0.0-1.0 |
| Semantic | Embedding similarity | 0.0-1.0 |

### Answer Formats

Supports common VQA answer formats:

- Yes/No answers
- Numbers
- Multiple choice (A/B/C/D)
- Short answers
- Long answers

## Specialized Verifiers

### VQAVerifier

Optimized for Visual Question Answering:

```python
from halo_forge.vlm.verifiers.base import VQAVerifier

verifier = VQAVerifier()  # Output weight = 0.6
```

### DocVQAVerifier

Optimized for document understanding:

```python
from halo_forge.vlm.verifiers.base import DocVQAVerifier

verifier = DocVQAVerifier()  # Perception weight = 0.4, OCR enabled
```

### ChartQAVerifier

Optimized for chart interpretation:

```python
from halo_forge.vlm.verifiers.base import ChartQAVerifier

verifier = ChartQAVerifier()  # Balanced weights
```

## Reward Calculation

The final reward is a weighted combination:

```
reward = (perception_weight × perception_score) +
         (reasoning_weight × reasoning_score) +
         (output_weight × output_score)
```

If no ground truth is provided, weights are redistributed:

```python
# Without ground truth
reward = (perception × perception_score + reasoning × reasoning_score) / 
         (perception + reasoning)
```

## Dependencies

Install perception verification dependencies:

```bash
pip install ultralytics easyocr
```

For semantic matching:

```bash
pip install sentence-transformers
```
