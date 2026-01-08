---
title: "Agentic Datasets"
description: "Tool calling datasets for training"
weight: 1
---

# Agentic / Tool Calling Datasets

halo forge supports several high-quality tool calling datasets.

---

## Available Datasets

### xLAM-60k (Primary)

**HuggingFace:** `Salesforce/xlam-function-calling-60k`

The gold standard for tool calling training:
- 60,000 verified samples
- 3,673 executable APIs
- >95% verification rate
- High-quality query-to-function mappings

```bash
halo-forge agentic benchmark --dataset xlam --limit 100
```

### Glaive Function Calling v2

**HuggingFace:** `glaiveai/glaive-function-calling-v2`

Large-scale dataset with irrelevance detection:
- 113,000 samples
- 7,500 irrelevance detection examples
- Good for training models to decline inappropriate requests

```bash
halo-forge agentic benchmark --dataset glaive --limit 100
```

### ToolBench

**HuggingFace:** `ToolBench/ToolBench`

Real-world API coverage:
- 188,000 samples
- 16,464 RESTful APIs
- Diverse real-world scenarios

---

## Dataset Composition

For optimal training, balance your dataset:

| Category | Target % | Description |
|----------|----------|-------------|
| Single tool call | 40% | Basic function calling |
| Parallel calls | 15% | Multiple tools in one response |
| Multi-turn chains | 20% | Tool result → followup → another tool |
| No tool needed | 15% | Irrelevance detection (critical!) |
| Error handling | 10% | Malformed input, failures |

---

## Loading Datasets

### Using CLI

```bash
# List available datasets
halo-forge agentic datasets

# Benchmark with specific dataset
halo-forge agentic benchmark --dataset xlam --limit 1000
```

### Using Python API

```python
from halo_forge.agentic.data import XLAMLoader, GlaiveLoader

# Load xLAM
loader = XLAMLoader()
samples = loader.load(limit=1000)

# Load Glaive with irrelevance samples
loader = GlaiveLoader()
samples = loader.load(limit=1000, include_irrelevant=True)

# Access sample data
for sample in samples[:3]:
    print(f"Query: {sample.messages[0]['content']}")
    print(f"Tools: {len(sample.tools)}")
    print(f"Expected: {sample.expected_calls}")
```

---

## Sample Structure

Each `ToolCallSample` contains:

```python
@dataclass
class ToolCallSample:
    messages: List[Dict]      # Conversation history
    tools: List[Dict]         # Available tool schemas
    expected_calls: List[Dict] # Expected tool_call outputs
    is_irrelevant: bool       # True if no tool should be called
    metadata: Dict            # Source, index, etc.
```

---

## Hermes Format Conversion

Convert samples to Hermes format for training:

```python
from halo_forge.agentic.data import HermesFormatter

formatter = HermesFormatter()

# Full training text
full = formatter.format(sample)

# Prompt only (for generation)
prompt = formatter.format_prompt(sample)
```

---

## Data Quality Tips

1. **Validate JSON** - Ensure all tool schemas and expected calls are valid JSON
2. **Include irrelevance** - At least 10% samples where no tool should be called
3. **Diverse tools** - Mix different API types and argument patterns
4. **Multi-turn** - Include examples with tool result → followup sequences
5. **Error cases** - Include examples of handling invalid inputs
