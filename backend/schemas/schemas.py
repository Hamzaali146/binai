from pydantic import BaseModel, Field
from typing import Optional, Any
from datetime import datetime
from enum import Enum


class TaskType(str, Enum):
    CLASSIFICATION = "classification"
    DETECTION = "detection"
    SEGMENTATION = "segmentation"


class EvaluationStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


# ── Model Registry ────────────────────────────────────────────────────────────

class ModelCreate(BaseModel):
    name: str
    version: str
    task_type: TaskType
    description: Optional[str] = None
    framework: Optional[str] = None
    model_path: Optional[str] = None
    config: dict = Field(default_factory=dict)


class ModelResponse(ModelCreate):
    id: str
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


# ── Datasets ──────────────────────────────────────────────────────────────────

class DatasetCreate(BaseModel):
    name: str
    task_type: TaskType
    description: Optional[str] = None
    path: str
    num_samples: int = 0
    num_classes: int = 0
    classes: list[str] = Field(default_factory=list)
    extra_metadata: dict = Field(default_factory=dict)


class DatasetResponse(DatasetCreate):
    id: str
    created_at: datetime

    class Config:
        from_attributes = True


# ── Evaluation ────────────────────────────────────────────────────────────────

class ScenarioConfig(BaseModel):
    name: str
    enabled: bool = True
    params: dict = Field(default_factory=dict)


class EvaluationConfig(BaseModel):
    confidence_threshold: float = 0.5
    iou_threshold: float = 0.5
    max_samples: Optional[int] = None
    scenarios: list[ScenarioConfig] = Field(default_factory=list)
    save_failures: bool = True
    generate_report: bool = True


class EvaluationCreate(BaseModel):
    name: str
    model_id: str
    dataset_id: str
    config: EvaluationConfig = Field(default_factory=EvaluationConfig)


class MetricsSummary(BaseModel):
    # Classification
    accuracy: Optional[float] = None
    precision: Optional[float] = None
    recall: Optional[float] = None
    f1_score: Optional[float] = None
    # Detection
    map_50: Optional[float] = None
    map_75: Optional[float] = None
    map_50_95: Optional[float] = None
    # Segmentation
    mean_iou: Optional[float] = None
    pixel_accuracy: Optional[float] = None
    # Common
    per_class_metrics: Optional[dict[str, Any]] = None


class ScenarioResult(BaseModel):
    scenario_name: str
    params: dict
    metrics: MetricsSummary
    num_samples: int
    num_failures: int
    degradation_pct: Optional[float] = None


class EvaluationResponse(BaseModel):
    id: str
    name: str
    model_id: str
    dataset_id: str
    task_type: TaskType
    status: EvaluationStatus
    config: dict
    metrics: Optional[MetricsSummary] = None
    scenario_results: list[ScenarioResult] = Field(default_factory=list)
    num_samples: int
    num_failures: int
    duration_seconds: Optional[float] = None
    error_message: Optional[str] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    created_at: datetime

    class Config:
        from_attributes = True


# ── Failure Cases ─────────────────────────────────────────────────────────────

class FailureCaseResponse(BaseModel):
    id: str
    evaluation_id: str
    image_id: Optional[str] = None
    image_path: Optional[str] = None
    ground_truth: Optional[Any] = None
    prediction: Optional[Any] = None
    confidence: Optional[float] = None
    iou_score: Optional[float] = None
    error_type: Optional[str] = None
    scenario: Optional[str] = None
    extra_metadata: dict = Field(default_factory=dict)
    created_at: datetime

    class Config:
        from_attributes = True


# ── Comparison ────────────────────────────────────────────────────────────────

class ModelComparisonRequest(BaseModel):
    evaluation_ids: list[str]
    metrics: list[str] = Field(default_factory=lambda: ["accuracy", "f1_score", "map_50"])


class ModelComparisonResponse(BaseModel):
    evaluations: list[dict]
    metric_deltas: dict[str, Any]
    best_model: Optional[str] = None


# ── Report ────────────────────────────────────────────────────────────────────

class ReportRequest(BaseModel):
    evaluation_id: str
    format: str = "json"    # json | pdf | html
    include_failures: bool = True
    include_scenarios: bool = True


class HealthResponse(BaseModel):
    status: str
    version: str
    timestamp: datetime
