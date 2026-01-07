---
title: "VLM Datasets"
weight: 20
---

# VLM Datasets

Dataset loaders for vision-language model training.

## Available Datasets

| Dataset | Task | Source | Samples |
|---------|------|--------|---------|
| TextVQA | Text reading | HuggingFace | 45K train |
| DocVQA | Document QA | HuggingFace | 50K train |
| ChartQA | Chart QA | HuggingFace | 28K train |
| RealWorldQA | Reasoning | HuggingFace | 700 test |
| MathVista | Math + Vision | HuggingFace | 6K+ test |

## Loading Datasets

### Using CLI

```bash
# List available datasets
halo-forge vlm datasets

# Train on dataset
halo-forge vlm train --dataset textvqa --model Qwen/Qwen2-VL-7B-Instruct

# Benchmark on dataset
halo-forge vlm benchmark --dataset docvqa --limit 100
```

### Using Python API

```python
from halo_forge.vlm.data import load_vlm_dataset, list_vlm_datasets

# List available
print(list_vlm_datasets())
# ['textvqa', 'docvqa', 'chartqa', 'realworldqa', 'mathvista']

# Load dataset
dataset = load_vlm_dataset("textvqa", split="train", limit=1000)

for sample in dataset:
    print(f"Prompt: {sample.prompt}")
    print(f"Ground truth: {sample.ground_truth}")
    image = sample.load_image()  # PIL Image
```

## Dataset Format

### VLMSample

Each sample contains:

```python
@dataclass
class VLMSample:
    image: Union[Image.Image, str]  # Image or path/URL
    prompt: str                      # Question/instruction
    ground_truth: Optional[str]      # Expected answer
    alternatives: Optional[List[str]] # Alternative answers
    metadata: Optional[Dict]         # Additional info
```

### Loading Images

```python
sample = dataset[0]

# Image might be PIL Image or path
if isinstance(sample.image, str):
    from PIL import Image
    img = Image.open(sample.image)
else:
    img = sample.image

# Or use the helper method
img = sample.load_image()
```

## TextVQA

Text reading and reasoning in natural images.

```python
from halo_forge.vlm.data import TextVQALoader

loader = TextVQALoader(split="train", limit=1000)
samples = loader.load()

print(f"Loaded {len(samples)} TextVQA samples")
```

**Sample format:**
- Image: Natural scene with text
- Prompt: Question about text in image
- Ground truth: Text answer
- Alternatives: Multiple valid answers

## DocVQA

Document understanding and information extraction.

```python
from halo_forge.vlm.data import DocVQALoader

loader = DocVQALoader(split="train", limit=1000)
samples = loader.load()
```

**Sample format:**
- Image: Document image (forms, receipts, etc.)
- Prompt: Question about document content
- Ground truth: Extracted information

## ChartQA

Chart interpretation and data extraction.

```python
from halo_forge.vlm.data import ChartQALoader

loader = ChartQALoader(split="train", limit=1000)
samples = loader.load()
```

**Sample format:**
- Image: Chart (bar, line, pie, etc.)
- Prompt: Question about data
- Ground truth: Numerical or descriptive answer
- Metadata: `type` (human or augmented)

## RealWorldQA

Real-world visual reasoning.

```python
from halo_forge.vlm.data import RealWorldQALoader

loader = RealWorldQALoader()  # Only test split available
samples = loader.load()
```

**Sample format:**
- Image: Real-world scene
- Prompt: Multiple choice question with options
- Ground truth: Correct choice (A/B/C/D)
- Metadata: `choices` list

## MathVista

Mathematical reasoning with visual context.

```python
from halo_forge.vlm.data import MathVistaLoader

loader = MathVistaLoader(split="testmini", limit=500)
samples = loader.load()
```

**Sample format:**
- Image: Math diagram, chart, or problem
- Prompt: Math question (may include choices)
- Ground truth: Answer
- Metadata: `question_type`, `grade`, `source`

## Exporting Datasets

### To RLVR Format

```python
dataset = load_vlm_dataset("textvqa", limit=1000)
dataset.to_rlvr_format("data/textvqa_rlvr.jsonl")
```

Output format:
```json
{"prompt": "...", "image": "/path/to/image.jpg", "ground_truth": "...", "metadata": {}}
```

### To SFT Format

```python
dataset = load_vlm_dataset("textvqa", limit=1000)
dataset.to_sft_format("data/textvqa_sft.jsonl", template="qwen")
```

Output format:
```json
{"text": "<|im_start|>user\n...<|im_end|>\n<|im_start|>assistant\n...<|im_end|>", "image": "..."}
```

## Custom Datasets

### From JSONL

```python
# Load custom dataset
trainer.train("path/to/custom_vlm.jsonl")
```

JSONL format:
```json
{"prompt": "What is in this image?", "image": "/path/to/img.jpg", "ground_truth": "A cat"}
{"prompt": "Read the text", "image": "/path/to/img2.jpg", "ground_truth": "Hello World"}
```

### Custom Loader

```python
from halo_forge.vlm.data.loaders import VLMDataset, VLMSample

class MyDataset(VLMDataset):
    @property
    def name(self) -> str:
        return "mydataset"
    
    def load(self):
        samples = []
        # Load your data
        for item in my_data:
            samples.append(VLMSample(
                image=item['image_path'],
                prompt=item['question'],
                ground_truth=item['answer']
            ))
        self.samples = samples
        return samples
```

## Dependencies

```bash
pip install datasets pillow
```

For image URLs:
```bash
pip install requests
```
