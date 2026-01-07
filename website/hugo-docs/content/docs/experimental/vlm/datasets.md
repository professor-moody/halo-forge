---
title: "VLM Datasets"
weight: 20
---

# VLM Datasets

Guide to obtaining and using datasets for Vision-Language Model training.

## Available Datasets

List available VLM datasets:

```bash
halo-forge vlm datasets
```

| Dataset | HuggingFace Path | Task | Size |
|---------|------------------|------|------|
| `textvqa` | `textvqa` | Text reading in images | 45K train |
| `docvqa` | `lmms-lab/DocVQA` | Document understanding | 50K train |
| `chartqa` | `HuggingFaceM4/ChartQA` | Chart interpretation | 28K train |
| `realworldqa` | `lmms-lab/RealWorldQA` | Real-world reasoning | 700 test |
| `mathvista` | `AI4Math/MathVista` | Mathematical reasoning | 6K+ test |

---

## Using Built-in Loaders

### Load from CLI

```bash
# Benchmark on TextVQA
halo-forge vlm benchmark \
  --model Qwen/Qwen2-VL-2B-Instruct \
  --dataset textvqa \
  --limit 100

# Train on DocVQA
halo-forge vlm train \
  --model Qwen/Qwen2-VL-2B-Instruct \
  --dataset docvqa \
  --cycles 4 \
  --output models/vlm_raft
```

### Load Programmatically

```python
from halo_forge.vlm.data import load_vlm_dataset, list_vlm_datasets

# List available datasets
print(list_vlm_datasets())
# ['textvqa', 'docvqa', 'chartqa', 'realworldqa', 'mathvista']

# Load TextVQA
dataset = load_vlm_dataset("textvqa", split="train", limit=1000)

# Iterate samples
for sample in dataset:
    print(f"Prompt: {sample.prompt}")
    print(f"Ground truth: {sample.ground_truth}")
    image = sample.load_image()  # PIL Image
```

---

## VLMSample Format

Each sample contains:

```python
@dataclass
class VLMSample:
    prompt: str           # Question/instruction
    image: str | Image    # Path, URL, or PIL Image
    ground_truth: str     # Expected answer
    metadata: Dict        # Additional info
```

### Example Sample

```python
sample = VLMSample(
    prompt="What text is shown on the sign?",
    image="path/to/image.jpg",
    ground_truth="STOP",
    metadata={"source": "textvqa", "difficulty": "easy"}
)
```

---

## Dataset Loaders

### TextVQA

Text reading in natural images (signs, labels, documents).

```python
from halo_forge.vlm.data.loaders import TextVQALoader

loader = TextVQALoader(split="train", limit=500)
samples = loader.load()

# Sample prompt: "What does the sign say?"
# Sample answer: "EXIT"
```

### DocVQA

Document understanding and information extraction.

```python
from halo_forge.vlm.data.loaders import DocVQALoader

loader = DocVQALoader(split="train", limit=500)
samples = loader.load()

# Sample prompt: "What is the total amount due?"
# Sample answer: "$1,234.56"
```

### ChartQA

Chart and graph interpretation.

```python
from halo_forge.vlm.data.loaders import ChartQALoader

loader = ChartQALoader(split="train", limit=500)
samples = loader.load()

# Sample prompt: "What is the value for Q3?"
# Sample answer: "150"
```

### RealWorldQA

Real-world reasoning from images.

```python
from halo_forge.vlm.data.loaders import RealWorldQALoader

loader = RealWorldQALoader(limit=200)
samples = loader.load()

# Sample prompt: "How many people are in the image?"
# Sample answer: "3"
```

### MathVista

Mathematical reasoning with visual context.

```python
from halo_forge.vlm.data.loaders import MathVistaLoader

loader = MathVistaLoader(split="test", limit=100)
samples = loader.load()

# Sample prompt: "What is the area of the shaded region?"
# Sample answer: "25 square units"
```

---

## Export Formats

### Export to RLVR Format

```python
dataset = load_vlm_dataset("textvqa", limit=1000)

# Export to JSONL
dataset.to_rlvr_format("vlm_rlvr.jsonl")
```

Output format:

```json
{
  "prompt": "What text is shown on the sign?",
  "image": "/path/to/image.jpg",
  "ground_truth": "STOP",
  "metadata": {"source": "textvqa"}
}
```

### Export to SFT Format

```python
dataset = load_vlm_dataset("docvqa", limit=1000)

# Export with Qwen template
dataset.to_sft_format("vlm_sft.jsonl", template="qwen")
```

Output format:

```json
{
  "messages": [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "<image>\nWhat is the total amount?"},
    {"role": "assistant", "content": "The total amount is $1,234.56"}
  ],
  "images": ["/path/to/image.jpg"]
}
```

---

## Creating Custom VLM Datasets

### From Local Images

```python
from halo_forge.vlm.data import VLMSample, VLMDataset
from pathlib import Path
import json

class CustomVLMDataset(VLMDataset):
    @property
    def name(self) -> str:
        return "custom"
    
    def load(self):
        # Load from annotation file
        with open("annotations.json") as f:
            data = json.load(f)
        
        self.samples = [
            VLMSample(
                prompt=item["question"],
                image=f"images/{item['image_id']}.jpg",
                ground_truth=item["answer"],
                metadata={"id": item["id"]}
            )
            for item in data
        ]
        return self.samples

# Use
dataset = CustomVLMDataset()
dataset.load()
dataset.to_rlvr_format("custom_vlm.jsonl")
```

### From HuggingFace

```python
from datasets import load_dataset
from halo_forge.vlm.data import VLMSample
import json

# Load any VQA dataset
hf_dataset = load_dataset("flaviagiammarino/vqa-rad", split="train")

# Convert to VLMSample format
samples = []
for item in hf_dataset:
    samples.append({
        "prompt": item["question"],
        "image": item["image"],  # PIL Image
        "ground_truth": item["answer"],
        "metadata": {"source": "vqa-rad"}
    })

# Save
with open("vqa_rad.jsonl", "w") as f:
    for s in samples:
        # Note: images need to be saved separately
        f.write(json.dumps({
            "prompt": s["prompt"],
            "ground_truth": s["ground_truth"],
            "metadata": s["metadata"]
        }) + "\n")
```

---

## HuggingFace Sources

### Recommended VLM Datasets

| Dataset | HuggingFace Path | Description |
|---------|------------------|-------------|
| TextVQA | `textvqa` | Text reading in images |
| DocVQA | `lmms-lab/DocVQA` | Document QA |
| ChartQA | `HuggingFaceM4/ChartQA` | Chart understanding |
| VQA v2 | `HuggingFaceM4/VQAv2` | General visual QA |
| OK-VQA | `Multimodal-Fatima/OK-VQA_train` | Knowledge-based VQA |
| GQA | `lmms-lab/GQA` | Compositional reasoning |
| AI2D | `lmms-lab/ai2d` | Science diagrams |
| InfoVQA | `lmms-lab/infographicvqa` | Infographic QA |

### Loading Directly

```python
from datasets import load_dataset

# Load VQA v2
vqa = load_dataset("HuggingFaceM4/VQAv2", split="train")

# Access samples
for item in vqa:
    question = item["question"]
    image = item["image"]  # PIL Image
    answers = item["answers"]
```

---

## Image Preprocessing

### VLMPreprocessor

```python
from halo_forge.vlm.data import VLMPreprocessor

processor = VLMPreprocessor(
    target_size=(224, 224),
    normalize=True
)

# Process image
result = processor.process("image.jpg")
pixel_values = result.pixel_values  # Tensor
```

### Custom Preprocessing

```python
from PIL import Image
import torchvision.transforms as T

transform = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

image = Image.open("image.jpg")
tensor = transform(image)
```

---

## Memory Considerations

VLM datasets can be memory-intensive:

| Dataset | Images | Typical Memory |
|---------|--------|----------------|
| TextVQA | 45K | ~20 GB disk |
| DocVQA | 50K | ~30 GB disk |
| ChartQA | 28K | ~15 GB disk |

### Tips

1. Use `limit` parameter for testing
2. Stream images instead of loading all at once
3. Use smaller image sizes for development
4. Clear cache: `rm -rf ~/.cache/halo_forge/vlm`

---

## Next Steps

- [VLM Training](./) - Train VLM models
- [VLM Testing](./testing.md) - Test VLM features
- [Code Datasets](../../training-pipeline/datasets-code/) - Code generation data
