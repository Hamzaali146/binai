from datetime import datetime
from sqlalchemy import (
    Column, String, Integer, Float, Boolean, DateTime,
    ForeignKey, JSON, Text, Enum as SAEnum
)
from sqlalchemy.orm import relationship
import enum
from backend.db.database import Base


class TaskType(str, enum.Enum):
    CLASSIFICATION = "classification"
    DETECTION = "detection"
    SEGMENTATION = "segmentation"


class EvaluationStatus(str, enum.Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class ModelRegistry(Base):
    __tablename__ = "model_registry"

    id = Column(String, primary_key=True)
    name = Column(String, nullable=False)
    version = Column(String, nullable=False)
    task_type = Column(SAEnum(TaskType), nullable=False)
    description = Column(Text, nullable=True)
    framework = Column(String, nullable=True)          # pytorch, tensorflow, onnx
    model_path = Column(String, nullable=True)
    config = Column(JSON, default={})
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    evaluations = relationship("EvaluationRun", back_populates="model")


class Dataset(Base):
    __tablename__ = "datasets"

    id = Column(String, primary_key=True)
    name = Column(String, nullable=False)
    task_type = Column(SAEnum(TaskType), nullable=False)
    description = Column(Text, nullable=True)
    path = Column(String, nullable=False)
    num_samples = Column(Integer, default=0)
    num_classes = Column(Integer, default=0)
    classes = Column(JSON, default=[])
    extra_metadata = Column(JSON, default={})
    created_at = Column(DateTime, default=datetime.utcnow)

    evaluations = relationship("EvaluationRun", back_populates="dataset")


class EvaluationRun(Base):
    __tablename__ = "evaluation_runs"

    id = Column(String, primary_key=True)
    name = Column(String, nullable=False)
    model_id = Column(String, ForeignKey("model_registry.id"), nullable=False)
    dataset_id = Column(String, ForeignKey("datasets.id"), nullable=False)
    task_type = Column(SAEnum(TaskType), nullable=False)
    status = Column(SAEnum(EvaluationStatus), default=EvaluationStatus.PENDING)
    config = Column(JSON, default={})
    metrics = Column(JSON, default={})
    scenario_results = Column(JSON, default={})
    num_samples = Column(Integer, default=0)
    num_failures = Column(Integer, default=0)
    duration_seconds = Column(Float, nullable=True)
    error_message = Column(Text, nullable=True)
    started_at = Column(DateTime, nullable=True)
    completed_at = Column(DateTime, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    model = relationship("ModelRegistry", back_populates="evaluations")
    dataset = relationship("Dataset", back_populates="evaluations")
    failure_cases = relationship("FailureCase", back_populates="evaluation")
    scenario_runs = relationship("ScenarioRun", back_populates="evaluation")


class ScenarioRun(Base):
    __tablename__ = "scenario_runs"

    id = Column(String, primary_key=True)
    evaluation_id = Column(String, ForeignKey("evaluation_runs.id"), nullable=False)
    scenario_name = Column(String, nullable=False)
    scenario_params = Column(JSON, default={})
    metrics = Column(JSON, default={})
    num_samples = Column(Integer, default=0)
    num_failures = Column(Integer, default=0)
    degradation_pct = Column(Float, nullable=True)   # % drop vs baseline
    created_at = Column(DateTime, default=datetime.utcnow)

    evaluation = relationship("EvaluationRun", back_populates="scenario_runs")


class FailureCase(Base):
    __tablename__ = "failure_cases"

    id = Column(String, primary_key=True)
    evaluation_id = Column(String, ForeignKey("evaluation_runs.id"), nullable=False)
    image_path = Column(String, nullable=True)
    image_id = Column(String, nullable=True)
    ground_truth = Column(JSON, nullable=True)
    prediction = Column(JSON, nullable=True)
    confidence = Column(Float, nullable=True)
    iou_score = Column(Float, nullable=True)
    error_type = Column(String, nullable=True)        # false_positive, false_negative, wrong_class
    scenario = Column(String, nullable=True)          # which scenario caused this failure
    extra_metadata = Column(JSON, default={})
    created_at = Column(DateTime, default=datetime.utcnow)

    evaluation = relationship("EvaluationRun", back_populates="failure_cases")
