"""
Results Service

Canonical ingestion and normalization of benchmark result files for UI consumers.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class BenchmarkResult:
    """Normalized benchmark result entry."""

    id: str
    model: str
    benchmark: str
    pass_at_1: Optional[float] = None
    pass_at_5: Optional[float] = None
    pass_at_10: Optional[float] = None
    accuracy: Optional[float] = None
    samples: int = 0
    duration_seconds: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)
    domain: str = "code"  # code, reasoning, vlm, audio, agentic
    notes: Optional[str] = None
    file_path: Optional[Path] = None
    launch_context_path: Optional[Path] = None
    has_relaunch_context: bool = False
    raw_data: Dict[str, Any] = field(default_factory=dict)
    normalized_metrics: Dict[str, float] = field(default_factory=dict)

    @property
    def primary_metric(self) -> Optional[float]:
        """Primary score used for summary cards/charts."""
        for key in ("pass_at_1", "accuracy", "avg_reward", "success_rate", "score"):
            if key in self.normalized_metrics:
                return self.normalized_metrics[key]
        if self.pass_at_1 is not None:
            return self.pass_at_1
        if self.accuracy is not None:
            return self.accuracy
        return None

    @property
    def primary_metric_name(self) -> str:
        """Name of the primary score."""
        for key, label in (
            ("pass_at_1", "pass@1"),
            ("accuracy", "accuracy"),
            ("avg_reward", "avg reward"),
            ("success_rate", "success rate"),
            ("score", "score"),
        ):
            if key in self.normalized_metrics:
                return label
        if self.pass_at_1 is not None:
            return "pass@1"
        if self.accuracy is not None:
            return "accuracy"
        return "score"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "id": self.id,
            "model": self.model,
            "benchmark": self.benchmark,
            "pass_at_1": self.pass_at_1,
            "pass_at_5": self.pass_at_5,
            "pass_at_10": self.pass_at_10,
            "accuracy": self.accuracy,
            "samples": self.samples,
            "duration_seconds": self.duration_seconds,
            "timestamp": self.timestamp.isoformat(),
            "domain": self.domain,
            "notes": self.notes,
            "launch_context_path": str(self.launch_context_path) if self.launch_context_path else None,
            "has_relaunch_context": self.has_relaunch_context,
            "normalized_metrics": self.normalized_metrics,
        }


@dataclass
class TrainingRunSummary:
    """Normalized modality training summary entry."""

    id: str
    modality: str
    model_name: str
    output_dir: Path
    timestamp: datetime = field(default_factory=datetime.now)
    run_id: Optional[str] = None
    seed: Optional[int] = None
    resume_from_cycle: int = 0
    resumed_from_checkpoint: Optional[Dict[str, Any]] = None
    base_model_name: Optional[str] = None
    active_model_name: Optional[str] = None
    cycles_executed: int = 0
    total_train_steps_executed: int = 0
    final_train_loss: Optional[float] = None
    weights_updated: bool = False
    final_update_reason: str = ""
    failure_reason: Optional[str] = None
    final_model_path: Optional[str] = None
    launch_context_path: Optional[Path] = None
    has_relaunch_context: bool = False
    cycle_losses: List[float] = field(default_factory=list)
    raw_data: Dict[str, Any] = field(default_factory=dict)


@dataclass
class UtilityRunSummary:
    """Normalized utility-module run summary entry."""

    id: str
    module: str
    execution_mode: str
    status: str
    return_code: int
    output_dir: Path
    run_summary_path: Path
    timestamp: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    duration_seconds: Optional[float] = None
    command: List[str] = field(default_factory=list)
    artifact_pointers: Dict[str, str] = field(default_factory=dict)
    launch_context_path: Optional[Path] = None
    has_relaunch_context: bool = False
    error_message: Optional[str] = None
    raw_data: Dict[str, Any] = field(default_factory=dict)


@dataclass
class QualificationReportSummary:
    """Normalized all-module qualification report entry."""

    id: str
    report_path: Path
    profile: str
    source: str
    status: str
    pass_count: int
    warn_count: int
    fail_count: int
    timestamp: datetime = field(default_factory=datetime.now)
    module_statuses: Dict[str, str] = field(default_factory=dict)
    failed_modules: List[str] = field(default_factory=list)
    module_issue_codes: Dict[str, str] = field(default_factory=dict)
    top_issue_code: Optional[str] = None
    top_fix_now: Optional[str] = None
    launch_context_path: Optional[Path] = None
    has_relaunch_context: bool = False
    raw_data: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BootstrapReportSummary:
    """Normalized all-module bootstrap report entry."""

    id: str
    report_path: Path
    profile: str
    source: str
    status: str
    pass_count: int
    warn_count: int
    fail_count: int
    timestamp: datetime = field(default_factory=datetime.now)
    module_statuses: Dict[str, str] = field(default_factory=dict)
    failed_modules: List[str] = field(default_factory=list)
    top_error: Optional[str] = None
    launch_context_path: Optional[Path] = None
    has_relaunch_context: bool = False
    raw_data: Dict[str, Any] = field(default_factory=dict)


class ResultsService:
    """Authoritative results ingestion/parsing and aggregation service."""

    RESULTS_DIRS = [
        Path("results"),
        Path("results/benchmarks"),
        Path("outputs"),
    ]
    TRAINING_DIRS = [
        Path("models"),
        Path("outputs"),
        Path("results"),
    ]
    UTILITY_DIRS = [
        Path("results/ops"),
    ]
    QUALIFICATION_DIRS = [
        Path("results/readiness"),
        Path("results/readiness/qualification"),
    ]
    BOOTSTRAP_DIRS = [
        Path("results/readiness"),
        Path("results/readiness/bootstrap"),
    ]

    DOMAIN_KEYWORDS = {
        "code": ["code", "humaneval", "mbpp", "livecodebench", "cpp", "rust", "go"],
        "reasoning": ["reasoning", "math", "gsm8k", "mmlu"],
        "vlm": ["vlm", "vision", "vqa", "textvqa", "docvqa", "chartqa"],
        "audio": ["audio", "speech", "asr", "librispeech", "common_voice", "commonvoice"],
        "agentic": ["agentic", "agent", "tool", "xlam", "function"],
    }

    DOMAIN_METRIC_COLUMNS: Dict[str, List[tuple[str, str]]] = {
        "code": [("pass_at_1", "pass@1"), ("pass_at_5", "pass@5"), ("pass_at_10", "pass@10")],
        "reasoning": [("accuracy", "Accuracy"), ("avg_reward", "Reward"), ("pass_at_1", "pass@1")],
        "vlm": [("accuracy", "Accuracy"), ("avg_reward", "Reward"), ("score", "Score")],
        "audio": [("success_rate", "Success"), ("wer", "WER"), ("avg_reward", "Reward")],
        "agentic": [
            ("accuracy", "Accuracy"),
            ("json_valid_rate", "JSON Valid"),
            ("function_correctness", "Fn Correct"),
        ],
    }

    _TRACKED_JSON_FILENAMES = {
        "benchmark.json",
        "summary.json",
    }

    def __init__(self, base_path: Optional[Path] = None):
        self.base_path = base_path or Path.cwd()
        self._cache: List[BenchmarkResult] = []
        self._cache_time: Optional[datetime] = None
        self._training_cache: List[TrainingRunSummary] = []
        self._training_cache_time: Optional[datetime] = None
        self._utility_cache: List[UtilityRunSummary] = []
        self._utility_cache_time: Optional[datetime] = None
        self._qualification_cache: List[QualificationReportSummary] = []
        self._qualification_cache_time: Optional[datetime] = None
        self._bootstrap_cache: List[BootstrapReportSummary] = []
        self._bootstrap_cache_time: Optional[datetime] = None
        self._cache_ttl = 30  # seconds

    def scan_results(self, force_refresh: bool = False) -> List[BenchmarkResult]:
        """Scan results directories for parseable benchmark JSON files."""
        if not force_refresh and self._cache_time:
            age = (datetime.now() - self._cache_time).total_seconds()
            if age < self._cache_ttl:
                return self._cache

        results: List[BenchmarkResult] = []
        seen_paths: set[Path] = set()

        for results_dir in self.RESULTS_DIRS:
            full_path = self.base_path / results_dir
            if not full_path.exists():
                continue

            for json_file in full_path.glob("**/*.json"):
                if json_file in seen_paths:
                    continue
                seen_paths.add(json_file)
                if not self._looks_like_result_file(json_file):
                    continue
                try:
                    parsed = self._parse_result_file(json_file)
                    if parsed:
                        results.append(parsed)
                except Exception as e:
                    print(f"[ResultsService] Failed to parse {json_file}: {e}")

        results.sort(key=lambda r: r.timestamp, reverse=True)
        self._cache = results
        self._cache_time = datetime.now()
        return results

    def list_results(self, force_refresh: bool = False) -> List[BenchmarkResult]:
        """Public alias for canonical listing."""
        return self.scan_results(force_refresh=force_refresh)

    def list_training_runs(self, force_refresh: bool = False) -> List[TrainingRunSummary]:
        """Scan canonical training summary artifacts for modality runs."""
        if not force_refresh and self._training_cache_time:
            age = (datetime.now() - self._training_cache_time).total_seconds()
            if age < self._cache_ttl:
                return self._training_cache

        runs: List[TrainingRunSummary] = []
        seen_files: set[Path] = set()

        for training_dir in self.TRAINING_DIRS:
            full_path = self.base_path / training_dir
            if not full_path.exists():
                continue

            for filename in ("training_summary.json", "training_metrics.json"):
                for json_file in full_path.glob(f"**/{filename}"):
                    if json_file in seen_files:
                        continue
                    seen_files.add(json_file)
                    if filename == "training_metrics.json" and (json_file.parent / "training_summary.json").exists():
                        continue
                    try:
                        parsed = self._parse_training_summary_file(json_file)
                        if parsed:
                            runs.append(parsed)
                    except Exception as e:
                        print(f"[ResultsService] Failed to parse training summary {json_file}: {e}")

        runs.sort(key=lambda r: r.timestamp, reverse=True)
        self._training_cache = runs
        self._training_cache_time = datetime.now()
        return runs

    def get_recent_training_runs(self, n: int = 5) -> List[TrainingRunSummary]:
        """Return the newest modality training runs."""
        return self.list_training_runs()[:n]

    def list_utility_runs(self, force_refresh: bool = False) -> List[UtilityRunSummary]:
        """Scan canonical utility run summaries for config/data/info/plot jobs."""
        if not force_refresh and self._utility_cache_time:
            age = (datetime.now() - self._utility_cache_time).total_seconds()
            if age < self._cache_ttl:
                return self._utility_cache

        runs: List[UtilityRunSummary] = []
        seen_files: set[Path] = set()

        for utility_dir in self.UTILITY_DIRS:
            full_path = self.base_path / utility_dir
            if not full_path.exists():
                continue

            for json_file in full_path.glob("**/run_summary.json"):
                if json_file in seen_files:
                    continue
                seen_files.add(json_file)
                try:
                    parsed = self._parse_utility_run_summary_file(json_file)
                    if parsed:
                        runs.append(parsed)
                except Exception as e:
                    print(f"[ResultsService] Failed to parse utility run summary {json_file}: {e}")

        runs.sort(key=lambda r: r.timestamp, reverse=True)
        self._utility_cache = runs
        self._utility_cache_time = datetime.now()
        return runs

    def get_recent_utility_runs(self, n: int = 10) -> List[UtilityRunSummary]:
        """Return newest utility module runs."""
        return self.list_utility_runs()[:n]

    def list_qualification_reports(
        self,
        force_refresh: bool = False,
    ) -> List[QualificationReportSummary]:
        """Scan canonical qualification report artifacts."""
        if not force_refresh and self._qualification_cache_time:
            age = (datetime.now() - self._qualification_cache_time).total_seconds()
            if age < self._cache_ttl:
                return self._qualification_cache

        reports: List[QualificationReportSummary] = []
        seen_files: set[Path] = set()

        for qualification_dir in self.QUALIFICATION_DIRS:
            full_path = self.base_path / qualification_dir
            if not full_path.exists():
                continue
            for json_file in full_path.glob("**/all_module_qualification.v1.json"):
                if json_file in seen_files:
                    continue
                seen_files.add(json_file)
                try:
                    parsed = self._parse_qualification_report_file(json_file)
                    if parsed:
                        reports.append(parsed)
                except Exception as e:
                    print(f"[ResultsService] Failed to parse qualification report {json_file}: {e}")

        reports.sort(key=lambda report: report.timestamp, reverse=True)
        self._qualification_cache = reports
        self._qualification_cache_time = datetime.now()
        return reports

    def get_recent_qualification_reports(self, n: int = 10) -> List[QualificationReportSummary]:
        """Return newest qualification report artifacts."""
        return self.list_qualification_reports()[:n]

    def list_bootstrap_reports(
        self,
        force_refresh: bool = False,
    ) -> List[BootstrapReportSummary]:
        """Scan canonical all-module bootstrap report artifacts."""
        if not force_refresh and self._bootstrap_cache_time:
            age = (datetime.now() - self._bootstrap_cache_time).total_seconds()
            if age < self._cache_ttl:
                return self._bootstrap_cache

        reports: List[BootstrapReportSummary] = []
        seen_files: set[Path] = set()

        for bootstrap_dir in self.BOOTSTRAP_DIRS:
            full_path = self.base_path / bootstrap_dir
            if not full_path.exists():
                continue
            for json_file in full_path.glob("**/all_module_bootstrap.v1.json"):
                if json_file in seen_files:
                    continue
                seen_files.add(json_file)
                try:
                    parsed = self._parse_bootstrap_report_file(json_file)
                    if parsed:
                        reports.append(parsed)
                except Exception as e:
                    print(f"[ResultsService] Failed to parse bootstrap report {json_file}: {e}")

        reports.sort(key=lambda report: report.timestamp, reverse=True)
        self._bootstrap_cache = reports
        self._bootstrap_cache_time = datetime.now()
        return reports

    def get_recent_bootstrap_reports(self, n: int = 10) -> List[BootstrapReportSummary]:
        """Return newest bootstrap report artifacts."""
        return self.list_bootstrap_reports()[:n]

    def get_dashboard_training_summary(self, max_runs: int = 3) -> Dict[str, Any]:
        """Build chart-ready training loss series from canonical training summaries."""
        runs = self.get_recent_training_runs(max_runs)
        if not runs:
            return {"runs": [], "steps": []}

        max_steps = max((len(run.cycle_losses) for run in runs), default=0)
        steps = [str(i + 1) for i in range(max_steps)]
        series: List[Dict[str, Any]] = []
        for run in runs:
            series.append(
                {
                    "name": f"{run.modality}:{Path(run.model_name).name[:12]}",
                    "loss": run.cycle_losses,
                }
            )
        return {"runs": series, "steps": steps}

    def list_results_by_domain(
        self,
        domain: str,
        force_refresh: bool = False,
    ) -> List[BenchmarkResult]:
        """List results filtered by a domain key."""
        return [r for r in self.scan_results(force_refresh=force_refresh) if r.domain == domain]

    def get_results_grouped_by_domain(self, force_refresh: bool = False) -> Dict[str, List[BenchmarkResult]]:
        """Return results grouped by domain for domain-specific UI tables."""
        grouped: Dict[str, List[BenchmarkResult]] = {}
        for result in self.scan_results(force_refresh=force_refresh):
            grouped.setdefault(result.domain, []).append(result)
        return grouped

    def get_dashboard_benchmark_summary(self, max_models: int = 5) -> Dict[str, Any]:
        """Aggregate latest domain scores per model for dashboard charting."""
        domain_order = ["code", "reasoning", "vlm", "audio", "agentic"]
        domain_labels = [self._display_domain_name(domain) for domain in domain_order]

        latest_by_model_domain: Dict[str, Dict[str, BenchmarkResult]] = {}
        for result in self.scan_results():
            model_key = Path(str(result.model)).name or str(result.model)
            latest_for_model = latest_by_model_domain.setdefault(model_key, {})
            existing = latest_for_model.get(result.domain)
            if existing is None or result.timestamp > existing.timestamp:
                latest_for_model[result.domain] = result

        ranked_models: List[tuple[str, Dict[str, BenchmarkResult]]] = []
        for model_key, domain_results in latest_by_model_domain.items():
            observed_domains = 0
            aggregate = 0.0
            for domain in domain_order:
                score = self._result_score_for_dashboard(domain_results.get(domain))
                if score is not None:
                    observed_domains += 1
                    aggregate += score
            if observed_domains == 0:
                continue
            ranked_models.append((model_key, domain_results))

        ranked_models.sort(
            key=lambda item: (
                sum(
                    1
                    for domain in domain_order
                    if self._result_score_for_dashboard(item[1].get(domain)) is not None
                ),
                sum(
                    self._result_score_for_dashboard(item[1].get(domain)) or 0.0
                    for domain in domain_order
                ),
            ),
            reverse=True,
        )

        models: List[Dict[str, Any]] = []
        for model_key, domain_results in ranked_models[:max_models]:
            scores = []
            for domain in domain_order:
                score = self._result_score_for_dashboard(domain_results.get(domain))
                scores.append(round(score, 1) if score is not None else 0.0)
            models.append({"name": model_key[:20], "scores": scores})

        return {"domains": domain_labels, "models": models}

    def get_domain_metric_columns(self, domain: str) -> List[tuple[str, str]]:
        """Metric columns for domain-specific result tables."""
        return self.DOMAIN_METRIC_COLUMNS.get(domain, [("score", "Score")])

    def _looks_like_result_file(self, path: Path) -> bool:
        """Lightweight file-name filter to skip obvious non-result artifacts."""
        name = path.name.lower()
        if name in self._TRACKED_JSON_FILENAMES:
            return True
        if "benchmark" in name or "result" in name or "metrics" in name:
            return True
        if "test" in name or "verify" in name or "baseline" in name:
            return True

        excluded_names = {
            "adapter_config.json",
            "tokenizer_config.json",
            "special_tokens_map.json",
            "tokenizer.json",
            "trainer_state.json",
            "config.json",
            "generation_config.json",
        }
        if name in excluded_names:
            return False

        path_str = str(path).lower()
        excluded_fragments = ("/checkpoint-", "/cycle_", "/adapter", "tensorboard")
        if any(fragment in path_str for fragment in excluded_fragments):
            return False

        return False

    def _parse_result_file(self, path: Path) -> Optional[BenchmarkResult]:
        with path.open(encoding="utf-8") as f:
            raw = json.load(f)

        if isinstance(raw, list):
            if not raw or not isinstance(raw[0], dict):
                return None
            data = raw[0]
        elif isinstance(raw, dict):
            data = raw
        else:
            return None

        sources = self._metric_sources(data)
        model = str(self._first_non_empty(sources, "model", "model_name", "model_path") or "unknown")
        benchmark = str(
            self._first_non_empty(sources, "benchmark", "dataset", "task", "suite")
            or (path.parent.name if path.name == "benchmark.json" else path.stem)
        )
        domain = self._detect_domain(path, data, benchmark)

        pass_at_1 = self._normalize_ratio(
            self._extract_pass_at_k_value(sources, 1)
            or self._as_float(self._first_non_empty(sources, "pass_at_1", "pass@1"))
        )
        pass_at_5 = self._normalize_ratio(
            self._extract_pass_at_k_value(sources, 5)
            or self._as_float(self._first_non_empty(sources, "pass_at_5", "pass@5"))
        )
        pass_at_10 = self._normalize_ratio(
            self._extract_pass_at_k_value(sources, 10)
            or self._as_float(self._first_non_empty(sources, "pass_at_10", "pass@10"))
        )

        accuracy = self._normalize_ratio(
            self._as_float(self._first_non_empty(sources, "accuracy", "acc"))
        )
        avg_reward = self._as_float(self._first_non_empty(sources, "avg_reward", "reward", "score"))
        if avg_reward is None:
            avg_reward = self._as_float(
                self._first_non_empty(sources, "average_reward")
            )
        success_rate = self._normalize_ratio(
            self._as_float(self._first_non_empty(sources, "success_rate", "pass_rate"))
        )
        wer = self._normalize_ratio(
            self._as_float(
                self._first_non_empty(sources, "wer", "word_error_rate", "average_wer")
            )
        )
        json_valid_rate = self._normalize_ratio(
            self._as_float(self._first_non_empty(sources, "json_valid_rate", "json_validity"))
        )
        function_correctness = self._normalize_ratio(
            self._as_float(
                self._first_non_empty(
                    sources,
                    "function_correctness",
                    "function_call_accuracy",
                    "function_accuracy",
                )
            )
        )

        samples = self._as_int(
            self._first_non_empty(sources, "total_samples", "samples", "total", "n_samples", "count")
        )
        if samples == 0 and isinstance(data.get("results"), list):
            samples = len(data["results"])

        duration_seconds = self._extract_duration_seconds(sources)
        timestamp = self._parse_timestamp(data, path)
        notes = self._extract_notes(data, path)

        normalized_metrics: Dict[str, float] = {}
        if pass_at_1 is not None:
            normalized_metrics["pass_at_1"] = pass_at_1
        if pass_at_5 is not None:
            normalized_metrics["pass_at_5"] = pass_at_5
        if pass_at_10 is not None:
            normalized_metrics["pass_at_10"] = pass_at_10
        if accuracy is not None:
            normalized_metrics["accuracy"] = accuracy
        if avg_reward is not None:
            normalized_metrics["avg_reward"] = avg_reward
        if success_rate is not None:
            normalized_metrics["success_rate"] = success_rate
        if wer is not None:
            normalized_metrics["wer"] = wer
        if json_valid_rate is not None:
            normalized_metrics["json_valid_rate"] = json_valid_rate
        if function_correctness is not None:
            normalized_metrics["function_correctness"] = function_correctness

        try:
            relative_id = path.resolve().relative_to(self.base_path.resolve()).as_posix()
        except Exception:
            relative_id = path.as_posix()
        result_id = relative_id.replace("/", "_")
        launch_context_path = path.parent / "launch_context.json"
        return BenchmarkResult(
            id=result_id,
            model=model,
            benchmark=benchmark,
            pass_at_1=pass_at_1,
            pass_at_5=pass_at_5,
            pass_at_10=pass_at_10,
            accuracy=accuracy,
            samples=samples,
            duration_seconds=duration_seconds,
            timestamp=timestamp,
            domain=domain,
            notes=notes,
            file_path=path,
            launch_context_path=launch_context_path if launch_context_path.exists() else None,
            has_relaunch_context=launch_context_path.exists(),
            raw_data=data,
            normalized_metrics=normalized_metrics,
        )

    def _parse_training_summary_file(self, path: Path) -> Optional[TrainingRunSummary]:
        with path.open(encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            return None

        modality = str(data.get("modality") or "").strip().lower()
        if not modality:
            modality = self._detect_domain(path, data, str(path.parent.name))

        model_name = str(
            data.get("base_model_name")
            or data.get("model_name")
            or data.get("active_model_name")
            or "unknown"
        )

        cycle_entries = data.get("cycles") if isinstance(data.get("cycles"), list) else []
        cycle_losses: List[float] = []
        for entry in cycle_entries:
            if not isinstance(entry, dict):
                continue
            loss_value = self._as_float(entry.get("train_loss"))
            if loss_value is not None:
                cycle_losses.append(loss_value)

        resume_from_cycle = self._as_int(data.get("resume_from_cycle"))
        total_steps = self._as_int(data.get("total_train_steps_executed"))
        final_loss = self._as_float(data.get("final_train_loss"))
        timestamp = self._parse_timestamp(data, path)
        final_model_path = data.get("final_model_path")
        run_id = data.get("run_id")
        seed = data.get("seed")
        try:
            if seed is not None:
                seed = int(seed)
        except (TypeError, ValueError):
            seed = None

        try:
            relative_id = path.resolve().relative_to(self.base_path.resolve()).as_posix()
        except Exception:
            relative_id = path.as_posix()
        launch_context_path = path.parent / "launch_context.json"

        return TrainingRunSummary(
            id=relative_id.replace("/", "_"),
            modality=modality or "unknown",
            model_name=model_name,
            output_dir=path.parent,
            timestamp=timestamp,
            run_id=str(run_id) if run_id else None,
            seed=seed,
            resume_from_cycle=resume_from_cycle,
            resumed_from_checkpoint=(
                data.get("resumed_from_checkpoint")
                if isinstance(data.get("resumed_from_checkpoint"), dict)
                else None
            ),
            base_model_name=(
                str(data.get("base_model_name"))
                if data.get("base_model_name")
                else None
            ),
            active_model_name=(
                str(data.get("active_model_name"))
                if data.get("active_model_name")
                else None
            ),
            cycles_executed=self._as_int(data.get("cycles_executed")),
            total_train_steps_executed=total_steps,
            final_train_loss=final_loss,
            weights_updated=bool(data.get("weights_updated", False)),
            final_update_reason=str(data.get("final_update_reason") or ""),
            failure_reason=(
                str(data.get("failure_reason"))
                if data.get("failure_reason") not in (None, "")
                else None
            ),
            final_model_path=str(final_model_path) if final_model_path else None,
            launch_context_path=launch_context_path if launch_context_path.exists() else None,
            has_relaunch_context=launch_context_path.exists(),
            cycle_losses=cycle_losses,
            raw_data=data,
        )

    def _parse_utility_run_summary_file(self, path: Path) -> Optional[UtilityRunSummary]:
        with path.open(encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            return None

        module = str(data.get("module") or "").strip().lower() or path.parent.parent.name.lower()
        execution_mode = str(data.get("execution_mode") or "contract").strip().lower()
        status = str(data.get("status") or "failed").strip().lower()
        return_code = self._as_int(data.get("return_code"))
        timestamp = self._parse_timestamp(data, path)
        started_at = None
        completed_at = None
        started_raw = data.get("started_at")
        completed_raw = data.get("completed_at")
        if isinstance(started_raw, str):
            try:
                started_at = datetime.fromisoformat(started_raw.replace("Z", "+00:00"))
            except Exception:
                started_at = None
        if isinstance(completed_raw, str):
            try:
                completed_at = datetime.fromisoformat(completed_raw.replace("Z", "+00:00"))
            except Exception:
                completed_at = None

        command = []
        if isinstance(data.get("command"), list):
            command = [str(item) for item in data.get("command") if isinstance(item, str)]
        artifact_pointers: Dict[str, str] = {}
        if isinstance(data.get("artifact_pointers"), dict):
            for key, value in data.get("artifact_pointers", {}).items():
                if isinstance(key, str) and value is not None:
                    artifact_pointers[key] = str(value)

        launch_context_path = path.parent / "launch_context.json"
        duration_seconds = self._as_float(data.get("duration_seconds"))

        try:
            relative_id = path.resolve().relative_to(self.base_path.resolve()).as_posix()
        except Exception:
            relative_id = path.as_posix()

        return UtilityRunSummary(
            id=relative_id.replace("/", "_"),
            module=module,
            execution_mode=execution_mode,
            status=status,
            return_code=return_code,
            output_dir=path.parent,
            run_summary_path=path,
            timestamp=timestamp,
            started_at=started_at,
            completed_at=completed_at,
            duration_seconds=duration_seconds,
            command=command,
            artifact_pointers=artifact_pointers,
            launch_context_path=launch_context_path if launch_context_path.exists() else None,
            has_relaunch_context=launch_context_path.exists(),
            error_message=str(data.get("error_message")) if data.get("error_message") else None,
            raw_data=data,
        )

    def _parse_qualification_report_file(self, path: Path) -> Optional[QualificationReportSummary]:
        with path.open(encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            return None

        modules = data.get("modules")
        if not isinstance(modules, dict):
            return None

        module_statuses: Dict[str, str] = {}
        module_issue_codes: Dict[str, str] = {}
        pass_count = 0
        warn_count = 0
        fail_count = 0
        failed_modules: List[str] = []
        top_issue_code: Optional[str] = None
        top_fix_now: Optional[str] = None
        for module, payload in modules.items():
            if not isinstance(payload, dict):
                continue
            status = str(payload.get("status") or "").strip().lower() or "warn"
            issue_code = str(payload.get("issue_code") or "").strip()
            fix_now = str(payload.get("fix_now") or "").strip()
            module_statuses[str(module)] = status
            if issue_code:
                module_issue_codes[str(module)] = issue_code
            if status == "pass":
                pass_count += 1
            elif status == "fail":
                fail_count += 1
                failed_modules.append(str(module))
                if top_issue_code is None and issue_code:
                    top_issue_code = issue_code
                if top_fix_now is None and fix_now:
                    top_fix_now = fix_now
            else:
                warn_count += 1
                if top_issue_code is None and issue_code:
                    top_issue_code = issue_code
                if top_fix_now is None and fix_now:
                    top_fix_now = fix_now

        overall_status = "pass"
        if fail_count > 0:
            overall_status = "fail"
        elif warn_count > 0:
            overall_status = "warn"

        timestamp = self._parse_timestamp(data, path)
        launch_context_path = path.parent / "launch_context.json"

        try:
            relative_id = path.resolve().relative_to(self.base_path.resolve()).as_posix()
        except Exception:
            relative_id = path.as_posix()

        return QualificationReportSummary(
            id=relative_id.replace("/", "_"),
            report_path=path,
            profile=str(data.get("profile") or "contract-v1"),
            source=str(data.get("source") or "script"),
            status=overall_status,
            pass_count=pass_count,
            warn_count=warn_count,
            fail_count=fail_count,
            timestamp=timestamp,
            module_statuses=module_statuses,
            failed_modules=failed_modules,
            module_issue_codes=module_issue_codes,
            top_issue_code=top_issue_code,
            top_fix_now=top_fix_now,
            launch_context_path=launch_context_path if launch_context_path.exists() else None,
            has_relaunch_context=launch_context_path.exists(),
            raw_data=data,
        )

    def _parse_bootstrap_report_file(self, path: Path) -> Optional[BootstrapReportSummary]:
        with path.open(encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            return None

        modules = data.get("modules")
        if not isinstance(modules, dict):
            return None

        module_statuses: Dict[str, str] = {}
        pass_count = 0
        warn_count = 0
        fail_count = 0
        failed_modules: List[str] = []
        top_error: Optional[str] = None

        for module, payload in modules.items():
            if not isinstance(payload, dict):
                continue
            status = str(payload.get("status") or "").strip().lower() or "warn"
            module_statuses[str(module)] = status
            if status == "pass":
                pass_count += 1
            elif status == "fail":
                fail_count += 1
                failed_modules.append(str(module))
                if top_error is None and payload.get("errors"):
                    first_error = payload.get("errors", [None])[0]
                    if first_error:
                        top_error = str(first_error)
            else:
                warn_count += 1
                if top_error is None and payload.get("warnings"):
                    first_warn = payload.get("warnings", [None])[0]
                    if first_warn:
                        top_error = str(first_warn)

        overall_status = "pass"
        if fail_count > 0:
            overall_status = "fail"
        elif warn_count > 0:
            overall_status = "warn"

        timestamp = self._parse_timestamp(data, path)
        launch_context_path = path.parent / "launch_context.json"

        try:
            relative_id = path.resolve().relative_to(self.base_path.resolve()).as_posix()
        except Exception:
            relative_id = path.as_posix()

        return BootstrapReportSummary(
            id=relative_id.replace("/", "_"),
            report_path=path,
            profile=str(data.get("profile") or "contract-v1"),
            source=str(data.get("source") or "script"),
            status=overall_status,
            pass_count=pass_count,
            warn_count=warn_count,
            fail_count=fail_count,
            timestamp=timestamp,
            module_statuses=module_statuses,
            failed_modules=failed_modules,
            top_error=top_error,
            launch_context_path=launch_context_path if launch_context_path.exists() else None,
            has_relaunch_context=launch_context_path.exists(),
            raw_data=data,
        )

    def get_latest_artifact_roots(self) -> Dict[str, str]:
        """Return best-known output roots discovered from parsed result artifacts."""
        mapping: Dict[str, str] = {}

        for run in self.list_training_runs():
            module = str(run.modality or "").strip().lower()
            if module and module not in mapping:
                mapping[module] = str(run.output_dir)

        for run in self.list_utility_runs():
            module = str(run.module or "").strip().lower()
            if module and module not in mapping:
                mapping[module] = str(run.output_dir)

        for result in self.list_results():
            if not result.file_path:
                continue
            domain = str(result.domain or "").strip().lower()
            parent = str(result.file_path.parent)
            if domain == "code" and "benchmark_code" not in mapping:
                mapping["benchmark_code"] = parent
            if domain in {"vlm", "audio", "reasoning", "agentic"}:
                if "benchmark_non_code" not in mapping:
                    mapping["benchmark_non_code"] = parent
                if "benchmark" not in mapping:
                    mapping["benchmark"] = parent

        for report in self.list_bootstrap_reports():
            modules = report.raw_data.get("modules")
            if not isinstance(modules, dict):
                continue
            for module, payload in modules.items():
                if not isinstance(payload, dict):
                    continue
                evidence_root = payload.get("evidence_root")
                if not evidence_root:
                    continue
                module_key = str(module).strip().lower()
                if module_key and module_key not in mapping:
                    mapping[module_key] = str(evidence_root)

        return mapping

    def _metric_sources(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Return ordered dictionaries that may carry metric values."""
        sources: List[Dict[str, Any]] = [data]
        metrics = data.get("metrics")
        if isinstance(metrics, dict):
            sources.append(metrics)

        baseline = data.get("baseline")
        if isinstance(baseline, dict):
            sources.append(baseline)
            baseline_metrics = baseline.get("metrics")
            if isinstance(baseline_metrics, dict):
                sources.append(baseline_metrics)

        return sources

    def _extract_pass_at_k_value(self, sources: List[Dict[str, Any]], k: int) -> Optional[float]:
        key_variants = (str(k), k)
        for source in sources:
            pass_at_k = source.get("pass_at_k")
            if isinstance(pass_at_k, dict):
                for key in key_variants:
                    if key in pass_at_k:
                        value = self._as_float(pass_at_k[key])
                        if value is not None:
                            return value
        return None

    def _extract_duration_seconds(self, sources: List[Dict[str, Any]]) -> float:
        direct = self._as_float(
            self._first_non_empty(
                sources,
                "duration_seconds",
                "duration",
                "total_time_sec",
                "total_time",
            )
        )
        if direct is not None:
            return direct

        for source in sources:
            timing = source.get("timing")
            if isinstance(timing, dict):
                timing_value = self._as_float(
                    timing.get("total_time_sec") or timing.get("total_time")
                )
                if timing_value is not None:
                    return timing_value
        return 0.0

    def _extract_notes(self, data: Dict[str, Any], path: Path) -> Optional[str]:
        note_value = data.get("notes")
        if note_value:
            return str(note_value)
        config_value = data.get("config")
        if config_value:
            return f"Config: {config_value}"
        try:
            return str(path.relative_to(self.base_path))
        except Exception:
            return str(path)

    def _first_non_empty(self, sources: List[Dict[str, Any]], *keys: str) -> Any:
        for source in sources:
            for key in keys:
                if key in source and source[key] not in (None, "", []):
                    return source[key]
        return None

    def _as_float(self, value: Any) -> Optional[float]:
        try:
            if value in (None, "", "nan"):
                return None
            return float(value)
        except (TypeError, ValueError):
            return None

    def _as_int(self, value: Any) -> int:
        try:
            if value in (None, ""):
                return 0
            return int(value)
        except (TypeError, ValueError):
            return 0

    def _normalize_ratio(self, value: Optional[float]) -> Optional[float]:
        """Normalize percentages to 0..1 range while preserving already-normalized values."""
        if value is None:
            return None
        if value > 1.0 and value <= 100.0:
            return value / 100.0
        return value

    def _detect_domain(self, path: Path, data: Dict[str, Any], benchmark: str) -> str:
        domain_value = data.get("domain")
        if isinstance(domain_value, str) and domain_value.lower() in self.DOMAIN_KEYWORDS:
            return domain_value.lower()

        text = f"{path} {benchmark}".lower()
        for domain, keywords in self.DOMAIN_KEYWORDS.items():
            if any(keyword in text for keyword in keywords):
                return domain
        return "code"

    def _parse_timestamp(self, data: Dict[str, Any], path: Path) -> datetime:
        for key in ("timestamp", "created_at", "completed_at"):
            value = data.get(key)
            if isinstance(value, str):
                try:
                    return datetime.fromisoformat(value.replace("Z", "+00:00"))
                except Exception:
                    continue
            if isinstance(value, (int, float)):
                try:
                    return datetime.fromtimestamp(value)
                except Exception:
                    continue

        try:
            return datetime.fromtimestamp(path.stat().st_mtime)
        except Exception:
            return datetime.now()

    def _display_domain_name(self, domain: str) -> str:
        if domain == "vlm":
            return "VLM"
        return domain.capitalize()

    def _result_score_for_dashboard(self, result: Optional[BenchmarkResult]) -> Optional[float]:
        if result is None:
            return None

        metric_order = [metric for metric, _ in self.get_domain_metric_columns(result.domain)]
        for key in metric_order:
            value = result.normalized_metrics.get(key)
            if value is None:
                continue
            if key == "wer":
                value = max(0.0, 1.0 - value)
            if key in {
                "pass_at_1",
                "pass_at_5",
                "pass_at_10",
                "accuracy",
                "success_rate",
                "json_valid_rate",
                "function_correctness",
                "wer",
            }:
                return max(0.0, min(100.0, value * 100.0))
            return max(0.0, min(100.0, value))

        primary = result.primary_metric
        if primary is None:
            return None
        if primary <= 1.0:
            return primary * 100.0
        return primary

    def get_latest_results(self, n: int = 5) -> List[BenchmarkResult]:
        return self.scan_results()[:n]

    def get_results_by_model(self, model: str) -> List[BenchmarkResult]:
        model_lower = model.lower()
        return [r for r in self.scan_results() if model_lower in r.model.lower()]

    def get_results_by_domain(self, domain: str) -> List[BenchmarkResult]:
        return [r for r in self.scan_results() if r.domain == domain]

    def get_results_by_benchmark(self, benchmark: str) -> List[BenchmarkResult]:
        bench_lower = benchmark.lower()
        return [r for r in self.scan_results() if bench_lower in r.benchmark.lower()]

    def get_summary(self) -> Dict[str, Any]:
        results = self.scan_results()
        domains: Dict[str, int] = {}
        models = set()
        benchmarks = set()
        for result in results:
            domains[result.domain] = domains.get(result.domain, 0) + 1
            models.add(result.model)
            benchmarks.add(result.benchmark)

        return {
            "total_results": len(results),
            "unique_models": len(models),
            "unique_benchmarks": len(benchmarks),
            "by_domain": domains,
            "latest_timestamp": results[0].timestamp if results else None,
        }

    def compare_results(self, result_ids: List[str]) -> Dict[str, Any]:
        selected = [r for r in self.scan_results() if r.id in result_ids]
        if not selected:
            return {"error": "No results found"}

        comparison = {
            "models": [r.model for r in selected],
            "benchmarks": [r.benchmark for r in selected],
            "metrics": {},
        }
        for result in selected:
            label = f"{Path(result.model).name[:15]}..."
            if result.pass_at_1 is not None:
                comparison.setdefault("metrics", {}).setdefault("pass@1", {})[label] = result.pass_at_1
            if result.accuracy is not None:
                comparison.setdefault("metrics", {}).setdefault("accuracy", {})[label] = result.accuracy
        return comparison


_service: Optional[ResultsService] = None


def get_results_service() -> ResultsService:
    """Get singleton service."""
    global _service
    if _service is None:
        _service = ResultsService()
    return _service
