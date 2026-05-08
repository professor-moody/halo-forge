"""halo_forge.training — cross-trainer surfaces.

Today this just hosts the intent-driven training-template registry.
The trainer modules themselves (sft/, raft/, dpo/, grpo/, rm/, vlm/,
audio/, reasoning/, agentic/) live as siblings of this package.
"""

from halo_forge.training.templates import (
    CATEGORIES,
    TEMPLATES,
    TrainingTemplate,
    cli_invocation,
    get_template,
    list_categories,
    list_templates,
)

__all__ = [
    "CATEGORIES",
    "TEMPLATES",
    "TrainingTemplate",
    "cli_invocation",
    "get_template",
    "list_categories",
    "list_templates",
]
