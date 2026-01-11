"""
Results Service

Scans the results directory for benchmark results and provides
formatted data for the UI Results page.
"""

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any


@dataclass
class BenchmarkResult:
    """A single benchmark result."""
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
    domain: str = "code"  # code, vlm, audio, agentic
    notes: Optional[str] = None
    file_path: Optional[Path] = None
    raw_data: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def primary_metric(self) -> Optional[float]:
        """Get the primary metric for display."""
        if self.pass_at_1 is not None:
            return self.pass_at_1
        if self.accuracy is not None:
            return self.accuracy
        return None
    
    @property
    def primary_metric_name(self) -> str:
        """Get name of primary metric."""
        if self.pass_at_1 is not None:
            return "pass@1"
        if self.accuracy is not None:
            return "accuracy"
        return "score"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            'id': self.id,
            'model': self.model,
            'benchmark': self.benchmark,
            'pass_at_1': self.pass_at_1,
            'pass_at_5': self.pass_at_5,
            'pass_at_10': self.pass_at_10,
            'accuracy': self.accuracy,
            'samples': self.samples,
            'duration_seconds': self.duration_seconds,
            'timestamp': self.timestamp.isoformat(),
            'domain': self.domain,
            'notes': self.notes,
        }


class ResultsService:
    """
    Service for loading and managing benchmark results.
    
    Scans the results directory structure:
        results/
        ├── code/
        │   └── model_name/
        │       └── benchmark.json
        ├── vlm/
        │   └── model_name/
        │       └── benchmark.json
        ├── audio/
        └── agentic/
    """
    
    # Directories to scan for results
    RESULTS_DIRS = [
        Path("results"),
        Path("results/benchmarks"),
        Path("outputs"),
    ]
    
    # Domain mappings based on directory structure
    DOMAIN_DIRS = {
        "code": ["code", "humaneval", "mbpp", "livecodebench"],
        "vlm": ["vlm", "vision", "vqa"],
        "audio": ["audio", "speech", "asr"],
        "agentic": ["agentic", "agent", "tool"],
    }
    
    def __init__(self, base_path: Optional[Path] = None):
        """
        Initialize results service.
        
        Args:
            base_path: Base path to search for results. Defaults to cwd.
        """
        self.base_path = base_path or Path.cwd()
        self._cache: List[BenchmarkResult] = []
        self._cache_time: Optional[datetime] = None
        self._cache_ttl = 30  # seconds
    
    def scan_results(self, force_refresh: bool = False) -> List[BenchmarkResult]:
        """
        Scan results directories for benchmark JSON files.
        
        Args:
            force_refresh: Force cache refresh
            
        Returns:
            List of BenchmarkResult sorted by timestamp descending
        """
        # Check cache
        if not force_refresh and self._cache_time:
            age = (datetime.now() - self._cache_time).total_seconds()
            if age < self._cache_ttl:
                return self._cache
        
        results = []
        
        for results_dir in self.RESULTS_DIRS:
            full_path = self.base_path / results_dir
            if not full_path.exists():
                continue
            
            for json_file in full_path.glob("**/*.json"):
                try:
                    result = self._parse_result_file(json_file)
                    if result:
                        results.append(result)
                except Exception as e:
                    print(f"Failed to parse {json_file}: {e}")
        
        # Sort by timestamp descending
        results.sort(key=lambda r: r.timestamp, reverse=True)
        
        # Update cache
        self._cache = results
        self._cache_time = datetime.now()
        
        return results
    
    def _parse_result_file(self, path: Path) -> Optional[BenchmarkResult]:
        """Parse a single result JSON file."""
        with open(path) as f:
            data = json.load(f)
        
        # Determine domain from path
        domain = self._detect_domain(path)
        
        # Extract common fields
        model = data.get('model', data.get('model_name', 'unknown'))
        benchmark = data.get('benchmark', data.get('dataset', path.stem))
        
        result = BenchmarkResult(
            id=f"{path.parent.name}_{path.stem}",
            model=model,
            benchmark=benchmark,
            domain=domain,
            file_path=path,
            raw_data=data,
        )
        
        # Parse pass@k metrics (code benchmarks)
        if 'pass_at_k' in data:
            pass_at = data['pass_at_k']
            result.pass_at_1 = pass_at.get('1', pass_at.get(1))
            result.pass_at_5 = pass_at.get('5', pass_at.get(5))
            result.pass_at_10 = pass_at.get('10', pass_at.get(10))
        elif 'pass@1' in data:
            result.pass_at_1 = data['pass@1']
            result.pass_at_5 = data.get('pass@5')
            result.pass_at_10 = data.get('pass@10')
        
        # Parse accuracy (VLM benchmarks)
        if 'accuracy' in data:
            result.accuracy = data['accuracy']
        elif 'avg_reward' in data:
            result.accuracy = data['avg_reward']
        
        # Parse sample count
        result.samples = data.get('total_samples', data.get('samples', 0))
        if 'results' in data and isinstance(data['results'], list):
            result.samples = len(data['results'])
        
        # Parse duration
        result.duration_seconds = data.get('duration', data.get('duration_seconds', 0))
        
        # Parse timestamp
        result.timestamp = self._parse_timestamp(data, path)
        
        # Notes
        if 'notes' in data:
            result.notes = data['notes']
        elif 'config' in data:
            result.notes = f"Config: {data['config']}"
        
        return result
    
    def _detect_domain(self, path: Path) -> str:
        """Detect domain from file path."""
        path_str = str(path).lower()
        
        for domain, keywords in self.DOMAIN_DIRS.items():
            for keyword in keywords:
                if keyword in path_str:
                    return domain
        
        return "code"  # Default
    
    def _parse_timestamp(self, data: dict, path: Path) -> datetime:
        """Extract timestamp from data or file."""
        if 'timestamp' in data:
            try:
                ts = data['timestamp']
                if isinstance(ts, str):
                    # Try ISO format
                    return datetime.fromisoformat(ts.replace('Z', '+00:00'))
                elif isinstance(ts, (int, float)):
                    return datetime.fromtimestamp(ts)
            except Exception:
                pass
        
        if 'created_at' in data:
            try:
                return datetime.fromisoformat(data['created_at'])
            except Exception:
                pass
        
        # Fall back to file modification time
        try:
            return datetime.fromtimestamp(path.stat().st_mtime)
        except Exception:
            return datetime.now()
    
    def get_latest_results(self, n: int = 5) -> List[BenchmarkResult]:
        """
        Get the N most recent results.
        
        Args:
            n: Number of results to return
            
        Returns:
            List of most recent results
        """
        results = self.scan_results()
        return results[:n]
    
    def get_results_by_model(self, model: str) -> List[BenchmarkResult]:
        """Get all results for a specific model."""
        results = self.scan_results()
        model_lower = model.lower()
        return [r for r in results if model_lower in r.model.lower()]
    
    def get_results_by_domain(self, domain: str) -> List[BenchmarkResult]:
        """Get all results for a specific domain."""
        results = self.scan_results()
        return [r for r in results if r.domain == domain]
    
    def get_results_by_benchmark(self, benchmark: str) -> List[BenchmarkResult]:
        """Get all results for a specific benchmark."""
        results = self.scan_results()
        bench_lower = benchmark.lower()
        return [r for r in results if bench_lower in r.benchmark.lower()]
    
    def get_summary(self) -> Dict[str, Any]:
        """
        Get summary statistics of all results.
        
        Returns:
            Dictionary with summary stats
        """
        results = self.scan_results()
        
        domains = {}
        models = set()
        benchmarks = set()
        
        for r in results:
            domains[r.domain] = domains.get(r.domain, 0) + 1
            models.add(r.model)
            benchmarks.add(r.benchmark)
        
        return {
            'total_results': len(results),
            'unique_models': len(models),
            'unique_benchmarks': len(benchmarks),
            'by_domain': domains,
            'latest_timestamp': results[0].timestamp if results else None,
        }
    
    def compare_results(self, result_ids: List[str]) -> Dict[str, Any]:
        """
        Compare multiple results.
        
        Args:
            result_ids: List of result IDs to compare
            
        Returns:
            Comparison data for charting
        """
        results = self.scan_results()
        selected = [r for r in results if r.id in result_ids]
        
        if not selected:
            return {'error': 'No results found'}
        
        comparison = {
            'models': [r.model for r in selected],
            'benchmarks': [r.benchmark for r in selected],
            'metrics': {},
        }
        
        # Collect metrics
        for r in selected:
            label = f"{r.model[:15]}..."
            if r.pass_at_1 is not None:
                if 'pass@1' not in comparison['metrics']:
                    comparison['metrics']['pass@1'] = {}
                comparison['metrics']['pass@1'][label] = r.pass_at_1
            if r.accuracy is not None:
                if 'accuracy' not in comparison['metrics']:
                    comparison['metrics']['accuracy'] = {}
                comparison['metrics']['accuracy'][label] = r.accuracy
        
        return comparison


# Singleton instance
_service: Optional[ResultsService] = None


def get_results_service() -> ResultsService:
    """Get the singleton results service."""
    global _service
    if _service is None:
        _service = ResultsService()
    return _service
