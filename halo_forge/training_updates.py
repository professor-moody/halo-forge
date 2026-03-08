"""
Shared helpers for lightweight supervised weight updates.
"""

from typing import Any, Dict, Iterable, List


def _chunk_texts(texts: List[str], batch_size: int) -> Iterable[List[str]]:
    """Yield fixed-size batches from a list of text samples."""
    for i in range(0, len(texts), batch_size):
        yield texts[i : i + batch_size]


def run_text_supervised_updates(
    *,
    model: Any,
    tokenizer: Any,
    texts: List[str],
    learning_rate: float,
    batch_size: int = 1,
    gradient_accumulation_steps: int = 1,
    max_steps: int = 8,
    max_length: int = 2048,
    max_grad_norm: float | None = None,
) -> Dict[str, Any]:
    """
    Run a minimal supervised update loop on text samples.
    """
    if not texts:
        return {
            "train_steps_executed": 0,
            "train_loss": None,
            "weights_updated": False,
            "update_reason": "no_samples",
            "optimizer_steps": 0,
            "skipped_batches_non_finite": 0,
        }

    if model is None or tokenizer is None:
        return {
            "train_steps_executed": 0,
            "train_loss": None,
            "weights_updated": False,
            "update_reason": "model_or_tokenizer_missing",
            "optimizer_steps": 0,
            "skipped_batches_non_finite": 0,
        }

    import torch

    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    optimizer.zero_grad(set_to_none=True)

    total_loss = 0.0
    optimizer_steps = 0
    micro_steps = 0
    grad_accum = max(1, gradient_accumulation_steps)
    per_step_limit = max(1, max_steps)
    last_loss_value = 0.0
    initial_loss_value = None
    skipped_batches_non_finite = 0

    for batch in _chunk_texts(texts, max(1, batch_size)):
        encoded = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        )

        if not hasattr(model, "device"):
            try:
                model_device = next(model.parameters()).device
            except StopIteration:
                model_device = torch.device("cpu")
        else:
            model_device = model.device

        input_ids = encoded["input_ids"].to(model_device)
        attention_mask = encoded.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(model_device)

        labels = input_ids.clone()
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )
        loss = outputs.loss
        if loss is None:
            continue
        if not torch.isfinite(loss).item():
            skipped_batches_non_finite += 1
            optimizer.zero_grad(set_to_none=True)
            continue

        if initial_loss_value is None:
            initial_loss_value = float(loss.detach().item())
        (loss / grad_accum).backward()
        last_loss_value = float(loss.detach().item())
        micro_steps += 1

        if micro_steps % grad_accum == 0:
            if max_grad_norm is not None and max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            optimizer_steps += 1
            total_loss += last_loss_value

            if optimizer_steps >= per_step_limit:
                break

    # Flush trailing gradients if accumulation did not align exactly.
    if micro_steps % grad_accum != 0:
        if max_grad_norm is not None and max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        optimizer_steps += 1
        total_loss += last_loss_value

    if optimizer_steps == 0:
        return {
            "train_steps_executed": 0,
            "train_loss": None,
            "initial_train_loss": initial_loss_value,
            "weights_updated": False,
            "update_reason": "no_optimizer_steps",
            "optimizer_steps": 0,
            "skipped_batches_non_finite": skipped_batches_non_finite,
        }

    return {
        "train_steps_executed": optimizer_steps,
        "train_loss": total_loss / optimizer_steps if total_loss else 0.0,
        "initial_train_loss": initial_loss_value,
        "weights_updated": True,
        "update_reason": "updated",
        "optimizer_steps": optimizer_steps,
        "skipped_batches_non_finite": skipped_batches_non_finite,
    }
