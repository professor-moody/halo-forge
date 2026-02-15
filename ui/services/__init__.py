"""
halo-forge UI Services

Backend services that connect the UI to actual halo-forge functionality.
These services provide the "wiring" between UI components and CLI commands.

Architecture Overview:
    
    ┌─────────────────────────────────────────────────────────────┐
    │                        UI Pages                              │
    │  Dashboard │ Training │ Monitor │ Results │ Datasets │ ...  │
    └────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
    ┌─────────────────────────────────────────────────────────────┐
    │                      UI Services                             │
    │  ┌───────────────┐ ┌───────────────┐ ┌───────────────────┐  │
    │  │TrainingService│ │HardwareMonitor│ │VLMEvalKitIntegra..│  │
    │  │               │ │               │ │                   │  │
    │  │ launch_sft()  │ │ get_stats()   │ │ run_benchmark()   │  │
    │  │ launch_raft() │ │ add_callback()│ │ list_benchmarks() │  │
    │  │ stop_job()    │ │ start/stop()  │ │                   │  │
    │  └───────────────┘ └───────────────┘ └───────────────────┘  │
    │  ┌───────────────┐ ┌───────────────┐ ┌───────────────────┐  │
    │  │VerifierService│ │ResultsService │ │ DatasetsService   │  │
    │  │               │ │               │ │                   │  │
    │  │ verify()      │ │ scan_results()│ │ list_datasets()   │  │
    │  │ get_verifiers│ │ get_summary() │ │ preview_dataset() │  │
    │  └───────────────┘ └───────────────┘ └───────────────────┘  │
    └────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
    ┌─────────────────────────────────────────────────────────────┐
    │                   halo-forge Core                            │
    │    CLI Commands │ Trainers │ Verifiers │ VLMEvalKit (dep)   │
    └─────────────────────────────────────────────────────────────┘

Usage:
    from ui.services import (
        TrainingService,
        get_hardware_monitor,
        get_verifier_service,
        get_results_service,
        get_datasets_service,
    )
    
    # Or use shortcuts
    from ui.services import (
        launch_sft_training,
        launch_raft_training,
        get_gpu_stats,
        run_vlm_benchmark,
    )
"""

# Hardware monitoring (existing)
from .hardware import (
    GPUStats,
    get_gpu_stats,
    get_gpu_summary,
)

# Training service
from .training_service import (
    TrainingService,
    TrainingMetrics,
)

# Metrics parser
from .metrics_parser import (
    MetricsParser,
)

# Verifier service
from .verifier_service import (
    VerifierService,
    VerifierType,
    VerifyResult,
    VerifierInfo,
    get_verifier_service,
)

# Results service
from .results_service import (
    ResultsService,
    BenchmarkResult,
    TrainingRunSummary,
    UtilityRunSummary,
    get_results_service,
)

# Datasets service
from .datasets_service import (
    DatasetsService,
    DatasetInfo,
    get_datasets_service,
)

# VLMEvalKit integration
from .vlmevalkit_integration import (
    VLMEvalKitIntegration,
    BenchmarkBackend,
    run_vlm_benchmark,
    is_vlmevalkit_available,
    list_vlm_benchmarks,
    list_vlm_models,
    should_use_vlmevalkit,
    get_integration as get_vlmevalkit_integration,
)

# Hardware monitor with callbacks
from .hardware_monitor import (
    HardwareMonitor,
    get_hardware_monitor,
    get_gpu_stats_sync,
)

# Event bus for real-time updates
from .event_bus import (
    EventBus,
    Event,
    EventType,
    get_event_bus,
    build_transition_payload,
)

# Benchmark service
from .benchmark_service import (
    BenchmarkService,
    BenchmarkType,
    BenchmarkPreset,
    get_benchmark_service,
    get_presets_for_type,
    CODE_PRESETS,
    VLM_PRESETS,
    AUDIO_PRESETS,
    REASONING_PRESETS,
    AGENTIC_PRESETS,
    ALL_PRESETS,
)
from .inference_service import (
    InferenceService,
    get_inference_service,
)
from .module_ops_service import (
    ModuleOpsService,
    get_module_ops_service,
)

# Launch contracts
from .launch_contracts import (
    LaunchContract,
    SFT_LAUNCH_CONTRACT,
    RAFT_LAUNCH_CONTRACT,
    BENCHMARK_LAUNCH_CONTRACT,
    INFERENCE_OPTIMIZE_LAUNCH_CONTRACT,
    INFERENCE_BENCHMARK_LAUNCH_CONTRACT,
    VLM_TRAIN_LAUNCH_CONTRACT,
    AUDIO_TRAIN_LAUNCH_CONTRACT,
    REASONING_TRAIN_LAUNCH_CONTRACT,
    AGENTIC_TRAIN_LAUNCH_CONTRACT,
    MODALITY_TRAIN_LAUNCH_CONTRACTS,
    MODULE_OPS_LAUNCH_CONTRACT,
    UTILITY_MODULE_TYPES,
    UTILITY_EXECUTION_MODES,
    UI_SUPPORTED_TRAINING_MODES,
    UI_DEFERRED_TRAINING_MODES,
    validate_launch_payload,
    ensure_local_path_exists_if_pathlike,
)
from .launch_context import (
    LaunchContextV1,
    CYCLE_BASED_TRAINING_JOB_TYPES,
    LAUNCH_CONTEXT_FILENAME,
    LAUNCH_CONTEXT_CONTRACT_VERSION,
    launch_context_path_for_output_dir,
    read_launch_context,
)
from .modality_readiness_service import (
    ModalityReadinessService,
    get_modality_readiness_service,
)
from .ops_readiness_service import (
    OpsReadinessService,
    get_ops_readiness_service,
)


# Convenience functions for common operations

async def launch_sft_training(state, **kwargs) -> str:
    """Launch SFT training (convenience wrapper)."""
    service = TrainingService(state)
    return await service.launch_sft(**kwargs)


async def launch_raft_training(state, **kwargs) -> str:
    """Launch RAFT training (convenience wrapper)."""
    service = TrainingService(state)
    return await service.launch_raft(**kwargs)


async def stop_training(state, job_id: str) -> bool:
    """Stop a training job (convenience wrapper)."""
    service = TrainingService(state)
    return await service.stop_job(job_id)


__all__ = [
    # Training
    'TrainingService',
    'TrainingMetrics',
    'launch_sft_training',
    'launch_raft_training',
    'stop_training',
    
    # Hardware
    'HardwareMonitor',
    'GPUStats',
    'get_hardware_monitor',
    'get_gpu_stats',
    'get_gpu_stats_sync',
    'get_gpu_summary',
    
    # VLMEvalKit
    'VLMEvalKitIntegration',
    'BenchmarkBackend',
    'run_vlm_benchmark',
    'is_vlmevalkit_available',
    'list_vlm_benchmarks',
    'list_vlm_models',
    'should_use_vlmevalkit',
    'get_vlmevalkit_integration',
    
    # Verifiers
    'VerifierService',
    'VerifierType',
    'VerifyResult',
    'VerifierInfo',
    'get_verifier_service',
    
    # Results
    'ResultsService',
    'BenchmarkResult',
    'TrainingRunSummary',
    'UtilityRunSummary',
    'get_results_service',
    
    # Datasets
    'DatasetsService',
    'DatasetInfo',
    'get_datasets_service',
    
    # Metrics
    'MetricsParser',
    
    # Event bus
    'EventBus',
    'Event',
    'EventType',
    'get_event_bus',
    'build_transition_payload',
    
    # Benchmark
    'BenchmarkService',
    'BenchmarkType',
    'BenchmarkPreset',
    'get_benchmark_service',
    'get_presets_for_type',
    'CODE_PRESETS',
    'VLM_PRESETS',
    'AUDIO_PRESETS',
    'REASONING_PRESETS',
    'AGENTIC_PRESETS',
    'ALL_PRESETS',

    # Inference
    'InferenceService',
    'get_inference_service',
    'ModuleOpsService',
    'get_module_ops_service',

    # Launch contracts
    'LaunchContract',
    'SFT_LAUNCH_CONTRACT',
    'RAFT_LAUNCH_CONTRACT',
    'BENCHMARK_LAUNCH_CONTRACT',
    'INFERENCE_OPTIMIZE_LAUNCH_CONTRACT',
    'INFERENCE_BENCHMARK_LAUNCH_CONTRACT',
    'VLM_TRAIN_LAUNCH_CONTRACT',
    'AUDIO_TRAIN_LAUNCH_CONTRACT',
    'REASONING_TRAIN_LAUNCH_CONTRACT',
    'AGENTIC_TRAIN_LAUNCH_CONTRACT',
    'MODALITY_TRAIN_LAUNCH_CONTRACTS',
    'MODULE_OPS_LAUNCH_CONTRACT',
    'UTILITY_MODULE_TYPES',
    'UTILITY_EXECUTION_MODES',
    'UI_SUPPORTED_TRAINING_MODES',
    'UI_DEFERRED_TRAINING_MODES',
    'validate_launch_payload',
    'ensure_local_path_exists_if_pathlike',

    # Launch context
    'LaunchContextV1',
    'CYCLE_BASED_TRAINING_JOB_TYPES',
    'LAUNCH_CONTEXT_FILENAME',
    'LAUNCH_CONTEXT_CONTRACT_VERSION',
    'launch_context_path_for_output_dir',
    'read_launch_context',

    # Non-code modality readiness
    'ModalityReadinessService',
    'get_modality_readiness_service',
    'OpsReadinessService',
    'get_ops_readiness_service',
]
