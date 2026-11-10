from pathlib import Path
from typing import Literal

import yaml
from pydantic import Field

from .contracts import Model, Task


class EnvironmentConfig(Model):
    kind: Literal["mock", "robocasa"] = "mock"
    source_path: str | None = None
    robosuite_path: str | None = None
    extra_python_paths: list[str] = Field(default_factory=list)
    library_path: str | None = None
    renderer: Literal["egl", "osmesa"] = "egl"
    rpc_timeout_seconds: float = Field(default=60, gt=0)
    python: str | None = None
    task_name: str = "PickPlaceToasterToCounter"
    instruction_source: Literal["config", "environment"] = "config"
    scene_split: Literal["pretrain", "target"] = "target"
    task_group: Literal["atomic_seen", "composite_seen", "composite_unseen"] | None = None
    seed: int = 7
    enable_depth: bool = True
    camera_size: int = Field(default=256, ge=32)
    ignore_native_horizon: bool = False
    native_subtask_checks: bool = False


class VLAConfig(Model):
    kind: Literal["mock", "pi05"] = "mock"
    endpoint: str = "ws://127.0.0.1:8010"
    timeout_seconds: float = Field(default=30, gt=0)
    action_layout: Literal["eef_first", "base_first"] = "eef_first"
    state_layout: Literal["eef_first", "base_first"] = "eef_first"
    binary_encoding: Literal["zero_one", "signed"] = "zero_one"
    clip_actions: bool = False
    image_size: int = Field(default=224, ge=32)
    expected_contract: str | None = None
    openpi_path: str = "/home/junwue/deps/openpi-robocasa"
    python: str = "/home/junwue/envs/pi05/bin/python"
    checkpoint: str | None = None
    norm_stats: str | None = None
    normalization: Literal["zscore", "quantile"] = "zscore"
    training_config: str = "pi05_pretrain_human300"
    sampling_seed: int | None = Field(default=None, ge=0, lt=2**32)


class FloraMonitorConfig(Model):
    window_steps: int = Field(default=200, ge=2)
    max_calls: int = Field(default=100, ge=1)
    low_score: int = Field(default=2, ge=1, le=3)
    high_score: int = Field(default=4, ge=4, le=5)
    low_checks: int = Field(default=2, ge=2)
    plateau_checks: int = Field(default=3, ge=2)
    drop_points: int = Field(default=2, ge=1, le=4)
    review_cooldown_steps: int = Field(default=150, ge=0)


class DecisionConfig(Model):
    kind: Literal["mock", "decision3", "flora"] = "mock"
    flora: FloraMonitorConfig = Field(default_factory=FloraMonitorConfig)
    endpoint: str = "http://127.0.0.1:8020/v1/systemone"
    model: str = "vllm-sr/d3-flash"
    revision: str | None = None
    timeout_seconds: float = Field(default=5, gt=0)
    interval_steps: int = Field(default=5, ge=1)
    max_unreviewed_steps: int = Field(default=5, ge=0)
    max_age_seconds: float = Field(default=10, gt=0)
    max_age_steps: int = Field(default=15, ge=1)
    window_frames: int = Field(default=5, ge=1, le=5)
    primary_camera: str | None = Field(default=None, min_length=1)
    history_cameras: list[str] = Field(default_factory=lambda: ["left", "right", "wrist"], min_length=1, max_length=4)
    context_mode: Literal["compact", "full"] = "compact"
    keep_threshold: float = Field(default=0.8, ge=0, le=1)
    min_margin: float = Field(default=0.1, ge=0, le=1)
    sampling: Literal["recent", "multiscale"] = "multiscale"
    history_seconds: list[float] = Field(default_factory=lambda: [10, 5, 2, 0.5, 0], min_length=1, max_length=5)
    latest_cameras: list[str] = Field(default_factory=lambda: ["wrist", "right"], max_length=2)


class PlannerConfig(Model):
    kind: Literal["mock", "mock_explorer", "astra", "codex"] = "mock"
    release_validated_proposal_on_budget: bool = False
    codex_command: str = "codex"
    max_tool_calls: int = Field(default=24, ge=1)
    review_tool_limit: int | None = Field(default=None, ge=1, le=24)
    admission_directory: str | None = None
    admission_slots: int = Field(default=3, ge=1, le=8)
    admission_wait_seconds: float = Field(default=240, gt=0)
    review_admission_seconds: float = Field(default=5, gt=0)
    service_retry_limit: int = Field(default=0, ge=0, le=2)
    service_backoff_seconds: float = Field(default=10, ge=0, le=60)
    compact_context: bool = False
    compact_numeric_history: bool = False
    deduplicate_experiences: bool = False
    historical_memory_mode: Literal['summary', 'full'] = 'summary'
    historical_memory_chars: int = Field(default=10000, ge=4000, le=32000)
    working_memory_mode: Literal['full', 'summary'] = 'full'
    working_memory_chars: int = Field(default=14000, ge=6000, le=32000)
    thread_max_turns: int = Field(default=0, ge=0)
    preserve_initial_instruction: bool = True
    window_frames: int = Field(default=3, ge=1, le=5)
    endpoint: str = "https://api.openai.com/v1/responses"
    model: str = "gpt-6-astra"
    api_key_env: str = "OPENAI_API_KEY"
    reasoning_effort: str = "high"
    timeout_seconds: float = Field(default=90, gt=0)
    max_calls: int = Field(default=30, ge=1)
    max_total_tokens: int | None = Field(default=None, ge=1)
    max_turns_per_intervention: int = Field(default=6, ge=1)
    max_output_tokens: int = Field(default=4096, ge=128)


class FloraConfig(Model):
    enabled: bool = False
    endpoint: str = "http://127.0.0.1:8030/v1/outcome"
    model: str = "nvidia/Flora-Rewarder"
    revision: str = "031299e46e452fc0cedb4e0975eb9fa95fc37783"
    python: str = "/home/junwue/envs/flora/bin/python"
    cameras: list[str] = Field(default_factory=lambda: ["left", "wrist"], min_length=1, max_length=4)
    timeout_seconds: float = Field(default=30, gt=0)
    max_calls: int = Field(default=6, ge=1)
    max_segment_steps: int = Field(default=3000, ge=2)
    max_segment_seconds: float = Field(default=120, gt=0)
    with_rationale: bool = True
    quantization: Literal["none", "int8"] = "none"
    gpu_memory_gib: int | None = Field(default=None, ge=8)


class RuntimeConfig(Model):
    experiment_label: str = "default"
    observation_compression_level: int = Field(default=6, ge=1, le=9)
    supervision_enabled: bool = True
    initial_planning: bool = False
    nonblocking_review: bool = True
    max_reviews: int = Field(default=20, ge=1)
    review_max_age_steps: int = Field(default=100, ge=1)
    review_max_age_seconds: float = Field(default=120, gt=0)
    progress_review_cooldown_steps: int = Field(default=0, ge=0)
    review_stall_recheck_displacement_m: float | None = Field(default=None, gt=0)
    continue_vla_on_review_budget: bool = False
    allowed_interventions: list[str] = Field(
        default_factory=lambda: [
            "continue",
            "refine",
            "switch",
            "replan",
            "code",
            "observe",
            "stop",
        ]
    )
    max_steps: int = Field(default=200, ge=1)
    max_wall_seconds: float | None = Field(default=300, gt=0)
    chunk_steps: int = Field(default=5, ge=1, le=10)
    max_interventions: int = Field(default=15, ge=1)
    max_no_progress_interventions: int = Field(default=3, ge=1)
    recovery_seconds: float = Field(default=15, gt=0)
    recovery_steps: int = Field(default=30, ge=1)
    recovery_attempts: int = Field(default=2, ge=1)
    recovery_tool_calls: int = Field(default=256, ge=1)
    recovery_observation_bytes: int = Field(default=64 << 20, ge=1024, le=1 << 30)
    recovery_program_versions: int = Field(default=24, ge=1)
    recovery_total_steps: int = Field(default=60, ge=1)
    recovery_total_seconds: float = Field(default=120, gt=0)
    stagnation_steps: int = Field(default=10, ge=2)
    stagnation_enabled: bool = True
    stagnation_scope: Literal["whole_robot", "eef_translation"] = "whole_robot"
    stagnation_seconds: float = Field(default=4.0, gt=0, le=15)
    stagnation_position_m: float = Field(default=0.002, gt=0)
    stagnation_rotation_rad: float = Field(default=0.02, gt=0)
    stagnation_gripper_m: float = Field(default=0.001, gt=0)
    stagnation_command_threshold: float = Field(default=0.05, ge=0, le=1)
    stagnation_cooldown_seconds: float = Field(default=4.0, gt=0)
    memory_mode: Literal["frozen", "accumulate"] = "frozen"
    skill_library: str = "skills"
    experience_library: str = "memory/experiences"
    task_memory_library: str | None = None  # default: sibling tasks/ next to experience_library
    audit_interval_steps: int = Field(default=200, ge=0)
    recovery_sandbox: Literal["landlock", "bubblewrap", "trusted"] = "landlock"


class Config(Model):
    environment: EnvironmentConfig = Field(default_factory=EnvironmentConfig)
    vla: VLAConfig = Field(default_factory=VLAConfig)
    decision: DecisionConfig = Field(default_factory=DecisionConfig)
    planner: PlannerConfig = Field(default_factory=PlannerConfig)
    flora: FloraConfig = Field(default_factory=FloraConfig)
    runtime: RuntimeConfig = Field(default_factory=RuntimeConfig)
    task: Task


def load_config(path):
    return Config.model_validate(yaml.safe_load(Path(path).read_text()))
