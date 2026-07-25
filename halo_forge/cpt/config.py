"""Configuration for continued causal pretraining (CPT)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Mapping, Optional

from halo_forge.cpt.packing import PACKING_ALGORITHM
from halo_forge.runtime_determinism import DEFAULT_TRAINING_SEED


@dataclass
class CPTConfig:
    """Shared configuration for Hugging Face and native MLX CPT.

    ``adaptation`` is intentionally not defaulted.  Continued pretraining can
    update either LoRA weights or every model weight, and silently selecting
    one changes both memory requirements and the resulting artifact.
    """

    adaptation: Optional[str] = None

    # Model/tokenizer identity.
    model_name: str = "Qwen/Qwen2.5-0.5B"
    model: Optional[str] = None
    model_revision: Optional[str] = None
    model_hash: Optional[str] = None
    tokenizer_revision: Optional[str] = None
    tokenizer_hash: Optional[str] = None
    trust_remote_code: bool = True
    attn_implementation: Optional[str] = None

    # Immutable corpus inputs.  Dataset Lab artifacts normally provide both
    # paths; direct launches may provide a train file and derive validation.
    train_file: Optional[str] = None
    validation_file: Optional[str] = None
    training_artifact_id: Optional[str] = None
    training_artifact_hash: Optional[str] = None
    expected_packing_plan_hash: Optional[str] = None
    validation_fraction: float = 0.05

    # Deterministic paragraph/EOS packing.
    max_sequence_length: int = 2048
    packing: str = PACKING_ALGORITHM
    budget_mode: str = "passes"
    target_tokens: Optional[int] = None
    corpus_passes: Optional[float] = 1.0

    # LoRA.  These fields are ignored only when adaptation="full".
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.0
    target_modules: Optional[List[str]] = None
    use_dora: bool = False
    use_rslora: bool = False
    init_lora_weights: str = "true"

    # Optional HF runtime quantization.  Full adaptation cannot update
    # bitsandbytes-quantized base weights and is rejected during validation.
    load_in_4bit: bool = False
    allow_quantization_fallback: bool = True

    # Optimization.
    output_dir: str = "models/cpt"
    batch_size: int = 1
    gradient_accumulation_steps: int = 8
    learning_rate: float = 2e-5
    warmup_ratio: float = 0.03
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    max_steps: Optional[int] = None
    optim: str = "adamw_torch"
    bf16: bool = True
    gradient_checkpointing: bool = True
    seed: int = DEFAULT_TRAINING_SEED

    # Checkpoint/evaluation cadence.
    save_steps: int = 500
    save_total_limit: int = 3
    eval_steps: int = 250
    logging_steps: int = 10

    # Compatibility alias accepted by internal callers that already use the
    # guided own-data wire name.  It is normalized into ``adaptation``.
    adaptation_mode: Optional[str] = None

    def __post_init__(self) -> None:
        if self.model is not None and str(self.model).strip():
            requested_model = str(self.model).strip()
            if self.model_name != "Qwen/Qwen2.5-0.5B" and self.model_name != requested_model:
                raise ValueError("model and model_name disagree")
            self.model_name = requested_model
        self.model = self.model_name

        explicit = [
            str(value).strip().lower()
            for value in (self.adaptation, self.adaptation_mode)
            if value is not None and str(value).strip()
        ]
        if not explicit:
            raise ValueError("CPT requires explicit adaptation='lora' or adaptation='full'")
        if len(set(explicit)) != 1:
            raise ValueError("adaptation and adaptation_mode disagree")
        adaptation = explicit[0]
        if adaptation not in {"lora", "full"}:
            raise ValueError("adaptation must be explicitly set to 'lora' or 'full'")
        self.adaptation = adaptation
        self.adaptation_mode = adaptation

        if int(self.max_sequence_length) < 2:
            raise ValueError("max_sequence_length must be at least 2")
        if str(self.packing).strip().lower() not in {
            PACKING_ALGORITHM,
            "paragraph_eos_non_overlap",
        }:
            raise ValueError(
                f"packing must be {PACKING_ALGORITHM!r}; overlapping/sliding packing is unsupported"
            )
        self.packing = PACKING_ALGORITHM

        mode = str(self.budget_mode).strip().lower()
        if mode not in {"tokens", "passes"}:
            raise ValueError("budget_mode must be 'tokens' or 'passes'")
        self.budget_mode = mode
        if mode == "tokens":
            if self.target_tokens is None or int(self.target_tokens) <= 0:
                raise ValueError("target_tokens is required for token-budget CPT")
            self.target_tokens = int(self.target_tokens)
            self.corpus_passes = None
        else:
            if self.corpus_passes is None or float(self.corpus_passes) <= 0:
                raise ValueError("corpus_passes is required for pass-budget CPT")
            self.corpus_passes = float(self.corpus_passes)
            self.target_tokens = None

        for name in (
            "batch_size",
            "gradient_accumulation_steps",
            "save_steps",
            "save_total_limit",
            "eval_steps",
            "logging_steps",
        ):
            if int(getattr(self, name)) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.max_steps is not None and int(self.max_steps) <= 0:
            raise ValueError("max_steps must be positive when provided")
        if not 0 <= float(self.validation_fraction) < 1:
            raise ValueError("validation_fraction must be in [0, 1)")
        if adaptation == "lora" and int(self.lora_r) <= 0:
            raise ValueError("lora_r must be positive for LoRA adaptation")
        if adaptation == "full" and self.load_in_4bit:
            raise ValueError("full CPT cannot update a 4-bit quantized base model")
        if self.target_modules is None:
            self.target_modules = [
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ]
        for name in (
            "training_artifact_id",
            "training_artifact_hash",
            "expected_packing_plan_hash",
        ):
            value = getattr(self, name)
            if value is None:
                continue
            normalized = str(value).strip()
            if not normalized:
                raise ValueError(f"{name} cannot be empty")
            setattr(self, name, normalized)

    @property
    def effective_batch_size(self) -> int:
        return int(self.batch_size) * int(self.gradient_accumulation_steps)

    @property
    def max_seq_length(self) -> int:
        """Compatibility alias used by existing trainer helpers."""

        return int(self.max_sequence_length)

    @classmethod
    def from_corpus_training_config(
        cls,
        value: Any,
        *,
        train_file: Optional[str] = None,
        validation_file: Optional[str] = None,
        **overrides: Any,
    ) -> "CPTConfig":
        """Translate the guided own-data transport contract into trainer config."""

        raw = value.to_dict() if hasattr(value, "to_dict") else dict(value)
        effective_batch_size = int(raw.pop("effective_batch_size", 1) or 1)
        raw.pop("dataset_version_id", None)
        model = raw.pop("model", None)
        output = raw.pop("output", None)
        fields = {
            "adaptation": raw.get("adaptation"),
            "model_name": model,
            "max_sequence_length": raw.get("max_sequence_length"),
            "packing": raw.get("packing", PACKING_ALGORITHM),
            "budget_mode": raw.get("budget_mode", "passes"),
            "target_tokens": raw.get("target_tokens"),
            "corpus_passes": raw.get("corpus_passes"),
            "learning_rate": raw.get("learning_rate") or cls.learning_rate,
            "batch_size": effective_batch_size,
            "gradient_accumulation_steps": 1,
            "seed": raw.get("seed", DEFAULT_TRAINING_SEED),
            "output_dir": output or cls.output_dir,
            "train_file": train_file,
            "validation_file": validation_file,
        }
        fields.update(overrides)
        return cls(**fields)

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "CPTConfig":
        values = dict(value)
        aliases = {
            "model": "model_name",
            "output": "output_dir",
            "max_seq_length": "max_sequence_length",
        }
        for source, destination in aliases.items():
            if source in values and destination not in values:
                values[destination] = values.pop(source)
        return cls(**values)

    @classmethod
    def from_yaml(cls, path: str) -> "CPTConfig":
        import yaml

        with open(path, encoding="utf-8") as handle:
            raw = yaml.safe_load(handle) or {}
        if not isinstance(raw, Mapping):
            raise ValueError("CPT YAML must contain an object")
        values: dict[str, Any] = {}
        for section in ("model", "data", "packing", "budget", "lora", "training"):
            section_value = raw.get(section)
            if isinstance(section_value, Mapping):
                values.update(section_value)
        values.update(
            {
                key: value
                for key, value in raw.items()
                if key not in {"model", "data", "packing", "budget", "lora", "training"}
            }
        )
        aliases = {
            "name": "model_name",
            "output": "output_dir",
            "rank": "lora_r",
            "r": "lora_r",
            "alpha": "lora_alpha",
            "dropout": "lora_dropout",
            "max_seq_length": "max_sequence_length",
        }
        for source, destination in aliases.items():
            if source in values and destination not in values:
                values[destination] = values.pop(source)
        allowed = cls.__dataclass_fields__
        return cls(**{key: value for key, value in values.items() if key in allowed})


__all__ = ["CPTConfig"]
