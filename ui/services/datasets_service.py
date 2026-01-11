"""
Datasets Service

Discovers and previews datasets available for training and benchmarking.
"""

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any, Iterator


@dataclass
class DatasetInfo:
    """Information about a dataset."""
    name: str
    path: Path
    samples: int
    size_mb: float
    columns: List[str] = field(default_factory=list)
    format: str = "jsonl"  # jsonl, json, parquet, csv
    domain: str = "code"   # code, vlm, audio, text
    description: Optional[str] = None
    modified_at: Optional[datetime] = None
    preview: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'name': self.name,
            'path': str(self.path),
            'samples': self.samples,
            'size_mb': self.size_mb,
            'columns': self.columns,
            'format': self.format,
            'domain': self.domain,
            'description': self.description,
            'modified_at': self.modified_at.isoformat() if self.modified_at else None,
        }


class DatasetsService:
    """
    Service for discovering and previewing datasets.
    
    Scans data directories for:
    - JSONL files (training data)
    - JSON files (benchmark data)
    - Parquet files (large datasets)
    """
    
    # Directories to scan
    DATA_DIRS = [
        Path("data"),
        Path("data/rlvr"),
        Path("data/samples"),
        Path("datasets"),
    ]
    
    # Domain detection keywords
    DOMAIN_KEYWORDS = {
        "code": ["code", "humaneval", "mbpp", "python", "cpp", "rust", "go"],
        "vlm": ["vlm", "vision", "image", "vqa", "textvqa", "docvqa"],
        "audio": ["audio", "speech", "asr", "librispeech", "whisper"],
        "text": ["text", "chat", "instruction", "sft"],
        "math": ["math", "gsm", "reasoning"],
    }
    
    # Supported file extensions
    EXTENSIONS = [".jsonl", ".json", ".parquet", ".csv"]
    
    def __init__(self, base_path: Optional[Path] = None):
        """
        Initialize datasets service.
        
        Args:
            base_path: Base path for scanning. Defaults to cwd.
        """
        self.base_path = base_path or Path.cwd()
        self._cache: List[DatasetInfo] = []
        self._cache_time: Optional[datetime] = None
        self._cache_ttl = 60  # seconds
    
    def list_datasets(self, force_refresh: bool = False) -> List[DatasetInfo]:
        """
        List all available datasets.
        
        Args:
            force_refresh: Force cache refresh
            
        Returns:
            List of DatasetInfo
        """
        # Check cache
        if not force_refresh and self._cache_time:
            age = (datetime.now() - self._cache_time).total_seconds()
            if age < self._cache_ttl:
                return self._cache
        
        datasets = []
        seen_paths = set()
        
        for data_dir in self.DATA_DIRS:
            full_path = self.base_path / data_dir
            if not full_path.exists():
                continue
            
            for ext in self.EXTENSIONS:
                for file_path in full_path.glob(f"**/*{ext}"):
                    if file_path in seen_paths:
                        continue
                    seen_paths.add(file_path)
                    
                    try:
                        info = self._get_dataset_info(file_path)
                        if info:
                            datasets.append(info)
                    except Exception as e:
                        print(f"Failed to process {file_path}: {e}")
        
        # Sort by modified time descending
        datasets.sort(
            key=lambda d: d.modified_at or datetime.min,
            reverse=True
        )
        
        # Update cache
        self._cache = datasets
        self._cache_time = datetime.now()
        
        return datasets
    
    def _get_dataset_info(self, path: Path) -> Optional[DatasetInfo]:
        """Get info about a dataset file."""
        try:
            # Get file size
            size_bytes = path.stat().st_size
            size_mb = size_bytes / (1024 * 1024)
            
            # Get modified time
            modified_at = datetime.fromtimestamp(path.stat().st_mtime)
            
            # Detect format
            format_type = path.suffix[1:]  # Remove dot
            
            # Count samples and get preview based on format
            if format_type == "jsonl":
                samples, columns, preview = self._parse_jsonl(path)
            elif format_type == "json":
                samples, columns, preview = self._parse_json(path)
            elif format_type == "parquet":
                samples, columns, preview = self._parse_parquet(path)
            elif format_type == "csv":
                samples, columns, preview = self._parse_csv(path)
            else:
                return None
            
            # Detect domain
            domain = self._detect_domain(path, columns)
            
            return DatasetInfo(
                name=path.stem,
                path=path,
                samples=samples,
                size_mb=round(size_mb, 2),
                columns=columns,
                format=format_type,
                domain=domain,
                modified_at=modified_at,
                preview=preview,
            )
            
        except Exception as e:
            print(f"Error parsing {path}: {e}")
            return None
    
    def _parse_jsonl(self, path: Path) -> tuple[int, List[str], Dict]:
        """Parse JSONL file for info."""
        sample_count = 0
        columns = []
        preview = {}
        
        with open(path, 'r', encoding='utf-8', errors='replace') as f:
            for i, line in enumerate(f):
                if i == 0:
                    # Get first sample for columns and preview
                    try:
                        preview = json.loads(line)
                        columns = list(preview.keys())
                    except json.JSONDecodeError:
                        pass
                sample_count += 1
        
        return sample_count, columns, preview
    
    def _parse_json(self, path: Path) -> tuple[int, List[str], Dict]:
        """Parse JSON file for info."""
        with open(path, 'r', encoding='utf-8', errors='replace') as f:
            data = json.load(f)
        
        if isinstance(data, list):
            sample_count = len(data)
            if data:
                preview = data[0]
                columns = list(preview.keys()) if isinstance(preview, dict) else []
            else:
                preview = {}
                columns = []
        elif isinstance(data, dict):
            # Could be a single sample or metadata
            sample_count = 1
            columns = list(data.keys())
            preview = data
        else:
            sample_count = 0
            columns = []
            preview = {}
        
        return sample_count, columns, preview
    
    def _parse_parquet(self, path: Path) -> tuple[int, List[str], Dict]:
        """Parse Parquet file for info."""
        try:
            import pyarrow.parquet as pq
            
            table = pq.read_table(path)
            sample_count = table.num_rows
            columns = table.column_names
            
            # Get first row as preview
            if sample_count > 0:
                preview = {col: str(table.column(col)[0]) for col in columns[:10]}
            else:
                preview = {}
            
            return sample_count, list(columns), preview
            
        except ImportError:
            # pyarrow not available
            return 0, [], {}
        except Exception:
            return 0, [], {}
    
    def _parse_csv(self, path: Path) -> tuple[int, List[str], Dict]:
        """Parse CSV file for info."""
        import csv
        
        sample_count = 0
        columns = []
        preview = {}
        
        with open(path, 'r', encoding='utf-8', errors='replace') as f:
            reader = csv.DictReader(f)
            columns = reader.fieldnames or []
            
            for i, row in enumerate(reader):
                if i == 0:
                    preview = dict(row)
                sample_count += 1
        
        return sample_count, columns, preview
    
    def _detect_domain(self, path: Path, columns: List[str]) -> str:
        """Detect dataset domain from path and columns."""
        path_str = str(path).lower()
        cols_str = " ".join(columns).lower()
        
        for domain, keywords in self.DOMAIN_KEYWORDS.items():
            for keyword in keywords:
                if keyword in path_str or keyword in cols_str:
                    return domain
        
        # Default based on column names
        if "prompt" in cols_str or "solution" in cols_str:
            return "code"
        if "image" in cols_str:
            return "vlm"
        if "audio" in cols_str:
            return "audio"
        
        return "text"
    
    def preview_dataset(
        self,
        path: Path,
        n: int = 5,
        offset: int = 0
    ) -> List[Dict[str, Any]]:
        """
        Get preview samples from a dataset.
        
        Args:
            path: Path to dataset file
            n: Number of samples to return
            offset: Starting offset
            
        Returns:
            List of sample dictionaries
        """
        samples = []
        path = Path(path)
        
        if not path.exists():
            return samples
        
        format_type = path.suffix[1:]
        
        if format_type == "jsonl":
            with open(path, 'r', encoding='utf-8', errors='replace') as f:
                for i, line in enumerate(f):
                    if i < offset:
                        continue
                    if i >= offset + n:
                        break
                    try:
                        samples.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
        
        elif format_type == "json":
            with open(path, 'r', encoding='utf-8', errors='replace') as f:
                data = json.load(f)
            if isinstance(data, list):
                samples = data[offset:offset + n]
            else:
                samples = [data]
        
        elif format_type == "parquet":
            try:
                import pyarrow.parquet as pq
                table = pq.read_table(path)
                df = table.slice(offset, n).to_pandas()
                samples = df.to_dict('records')
            except Exception:
                pass
        
        elif format_type == "csv":
            import csv
            with open(path, 'r', encoding='utf-8', errors='replace') as f:
                reader = csv.DictReader(f)
                for i, row in enumerate(reader):
                    if i < offset:
                        continue
                    if i >= offset + n:
                        break
                    samples.append(dict(row))
        
        return samples
    
    def get_dataset_by_name(self, name: str) -> Optional[DatasetInfo]:
        """Find a dataset by name."""
        datasets = self.list_datasets()
        name_lower = name.lower()
        
        for ds in datasets:
            if ds.name.lower() == name_lower:
                return ds
        
        # Try partial match
        for ds in datasets:
            if name_lower in ds.name.lower():
                return ds
        
        return None
    
    def get_datasets_by_domain(self, domain: str) -> List[DatasetInfo]:
        """Get all datasets for a specific domain."""
        datasets = self.list_datasets()
        return [ds for ds in datasets if ds.domain == domain]
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary statistics of all datasets."""
        datasets = self.list_datasets()
        
        total_samples = sum(ds.samples for ds in datasets)
        total_size = sum(ds.size_mb for ds in datasets)
        
        by_domain = {}
        by_format = {}
        
        for ds in datasets:
            by_domain[ds.domain] = by_domain.get(ds.domain, 0) + 1
            by_format[ds.format] = by_format.get(ds.format, 0) + 1
        
        return {
            'total_datasets': len(datasets),
            'total_samples': total_samples,
            'total_size_mb': round(total_size, 2),
            'by_domain': by_domain,
            'by_format': by_format,
        }
    
    def iter_dataset(
        self,
        path: Path,
        batch_size: int = 100,
    ) -> Iterator[List[Dict[str, Any]]]:
        """
        Iterate over a dataset in batches.
        
        Useful for processing large datasets without loading all into memory.
        
        Args:
            path: Path to dataset file
            batch_size: Number of samples per batch
            
        Yields:
            Batches of sample dictionaries
        """
        path = Path(path)
        if not path.exists():
            return
        
        offset = 0
        while True:
            batch = self.preview_dataset(path, n=batch_size, offset=offset)
            if not batch:
                break
            yield batch
            offset += len(batch)
            if len(batch) < batch_size:
                break


# Singleton instance
_service: Optional[DatasetsService] = None


def get_datasets_service() -> DatasetsService:
    """Get the singleton datasets service."""
    global _service
    if _service is None:
        _service = DatasetsService()
    return _service
