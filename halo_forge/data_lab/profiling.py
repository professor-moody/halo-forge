"""Dependency-light dataset profiling for text, image, and audio records."""

from __future__ import annotations

import importlib
import statistics
import wave
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence


def _summary(values: Sequence[float | int]) -> Dict[str, Any]:
    if not values:
        return {"count": 0, "min": None, "max": None, "mean": None, "median": None}
    return {
        "count": len(values),
        "min": min(values),
        "max": max(values),
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
    }


def record_text(record: Mapping[str, Any], field: Optional[str] = None) -> str:
    if field:
        from .models import get_field

        value = get_field(record, field)
        return value if isinstance(value, str) else str(value or "")
    parts: List[str] = []
    for key in (
        "system",
        "prompt",
        "response",
        "reference_answer",
        "chosen",
        "rejected",
        "task",
        "transcript",
        "label",
        "ground_truth",
        "text",
        "completion",
    ):
        value = record.get(key)
        if isinstance(value, str):
            parts.append(value)
    messages = record.get("messages")
    if isinstance(messages, list):
        parts.extend(
            str(message.get("content", "")) for message in messages if isinstance(message, Mapping)
        )
    return " ".join(part for part in parts if part)


def text_profile(
    records: Sequence[Mapping[str, Any]], *, field: Optional[str] = None
) -> Dict[str, Any]:
    texts = [record_text(record, field) for record in records]
    characters = [len(text) for text in texts]
    words = [len(text.split()) for text in texts]
    empty = sum(not text.strip() for text in texts)
    languages: Dict[str, int] = {}
    # A cheap, transparent script distribution is more useful than pretending
    # this dependency-light profile is a full language detector.
    for text in texts:
        if any("\u4e00" <= char <= "\u9fff" for char in text):
            script = "cjk"
        elif any("\u0400" <= char <= "\u04ff" for char in text):
            script = "cyrillic"
        elif any("\u0600" <= char <= "\u06ff" for char in text):
            script = "arabic"
        else:
            script = "latin_or_other"
        languages[script] = languages.get(script, 0) + 1
    return {
        "rows": len(records),
        "empty_rows": empty,
        "characters": _summary(characters),
        "words": _summary(words),
        # Common tokenizer rule-of-thumb; intentionally labeled an estimate.
        "estimated_tokens": _summary([max(0, round(length / 4)) for length in characters]),
        "script_distribution": languages,
    }


def _references(record: Mapping[str, Any], fields: Sequence[str]) -> Iterable[str]:
    for field in fields:
        value = record.get(field)
        values = value if isinstance(value, list) else [value]
        for item in values:
            if isinstance(item, (str, Path)):
                yield str(item)
            elif isinstance(item, Mapping) and isinstance(item.get("path"), str):
                yield item["path"]


def _resolve(reference: str, base_dir: Optional[Path]) -> Optional[Path]:
    if reference.startswith(("http://", "https://", "data:")):
        return None
    path = Path(reference).expanduser()
    if not path.is_absolute() and base_dir is not None:
        path = base_dir / path
    return path.resolve()


def image_profile(
    records: Sequence[Mapping[str, Any]], *, base_dir: Optional[Path | str] = None
) -> Dict[str, Any]:
    root = Path(base_dir).expanduser().resolve() if base_dir else None
    refs = list(
        dict.fromkeys(ref for row in records for ref in _references(row, ("image", "image_path")))
    )
    widths: List[int] = []
    heights: List[int] = []
    formats: Dict[str, int] = {}
    missing = remote = unreadable = 0
    try:
        image_module = importlib.import_module("PIL.Image")
    except ImportError:
        image_module = None
    for reference in refs:
        path = _resolve(reference, root)
        if path is None:
            remote += 1
            continue
        if not path.is_file():
            missing += 1
            continue
        if image_module is None:
            formats[path.suffix.lower().lstrip(".") or "unknown"] = (
                formats.get(path.suffix.lower().lstrip(".") or "unknown", 0) + 1
            )
            continue
        try:
            with image_module.open(path) as image:
                widths.append(image.width)
                heights.append(image.height)
                fmt = str(image.format or path.suffix.lstrip(".") or "unknown").lower()
                formats[fmt] = formats.get(fmt, 0) + 1
        except Exception:
            unreadable += 1
    return {
        "references": len(refs),
        "missing": missing,
        "remote": remote,
        "unreadable": unreadable,
        "width": _summary(widths),
        "height": _summary(heights),
        "formats": formats,
    }


def _audio_info(path: Path) -> tuple[float, int, int]:
    if path.suffix.lower() in {".wav", ".wave"}:
        with wave.open(str(path), "rb") as handle:
            rate = handle.getframerate()
            frames = handle.getnframes()
            return frames / rate if rate else 0.0, rate, handle.getnchannels()
    try:
        soundfile = importlib.import_module("soundfile")
    except ImportError as exc:
        raise RuntimeError("non-WAV profiling requires soundfile") from exc
    info = soundfile.info(str(path))
    return float(info.duration), int(info.samplerate), int(info.channels)


def audio_profile(
    records: Sequence[Mapping[str, Any]], *, base_dir: Optional[Path | str] = None
) -> Dict[str, Any]:
    root = Path(base_dir).expanduser().resolve() if base_dir else None
    refs = list(
        dict.fromkeys(ref for row in records for ref in _references(row, ("audio", "audio_path")))
    )
    durations: List[float] = []
    sample_rates: Dict[str, int] = {}
    channels: Dict[str, int] = {}
    formats: Dict[str, int] = {}
    missing = remote = unreadable = 0
    for reference in refs:
        path = _resolve(reference, root)
        if path is None:
            remote += 1
            continue
        if not path.is_file():
            missing += 1
            continue
        fmt = path.suffix.lower().lstrip(".") or "unknown"
        formats[fmt] = formats.get(fmt, 0) + 1
        try:
            duration, rate, channel_count = _audio_info(path)
            durations.append(duration)
            sample_rates[str(rate)] = sample_rates.get(str(rate), 0) + 1
            channels[str(channel_count)] = channels.get(str(channel_count), 0) + 1
        except Exception:
            unreadable += 1
    return {
        "references": len(refs),
        "missing": missing,
        "remote": remote,
        "unreadable": unreadable,
        "duration_seconds": _summary(durations),
        "total_duration_seconds": sum(durations),
        "sample_rates": sample_rates,
        "channels": channels,
        "formats": formats,
    }


def profile_records(
    records: Sequence[Mapping[str, Any]], *, base_dir: Optional[Path | str] = None
) -> Dict[str, Any]:
    keys: Dict[str, int] = {}
    for row in records:
        for key in row:
            keys[key] = keys.get(key, 0) + 1
    result = {"row_count": len(records), "fields": keys, "text": text_profile(records)}
    if any(any(field in row for field in ("image", "image_path")) for row in records):
        result["image"] = image_profile(records, base_dir=base_dir)
    if any(any(field in row for field in ("audio", "audio_path")) for row in records):
        result["audio"] = audio_profile(records, base_dir=base_dir)
    return result


__all__ = ["audio_profile", "image_profile", "profile_records", "record_text", "text_profile"]
