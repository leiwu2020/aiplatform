from pydantic import BaseModel
from typing import Optional, Dict, Any, List
from datetime import datetime

# User schemas
class UserBase(BaseModel):
    username: str
    email: str

class UserCreate(UserBase):
    password: str

class User(UserBase):
    id: int
    is_active: bool
    created_at: datetime
    
    class Config:
        from_attributes = True

# Dataset schemas
class DatasetBase(BaseModel):
    name: str
    description: Optional[str] = None

class DatasetCreate(DatasetBase):
    pass

class Dataset(DatasetBase):
    id: int
    file_path: str
    file_size: int
    file_type: str
    status: str
    metadata: Optional[Dict[str, Any]] = None
    created_at: datetime
    owner_id: int
    
    class Config:
        from_attributes = True

# Model schemas
class ModelBase(BaseModel):
    name: str
    description: Optional[str] = None
    model_type: str
    base_model: Optional[str] = None

class ModelCreate(ModelBase):
    dataset_id: int
    config: Optional[Dict[str, Any]] = None

class ModelUpdate(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    config: Optional[Dict[str, Any]] = None

class Model(ModelBase):
    id: int
    status: str
    model_path: Optional[str] = None
    config: Optional[Dict[str, Any]] = None
    metrics: Optional[Dict[str, Any]] = None
    created_at: datetime
    updated_at: datetime
    owner_id: int
    dataset_id: int
    
    class Config:
        from_attributes = True

# Deployment schemas
class DeploymentBase(BaseModel):
    name: str
    replicas: int = 1
    cpu_limit: Optional[str] = None
    memory_limit: Optional[str] = None
    gpu_limit: Optional[str] = None

class DeploymentCreate(DeploymentBase):
    model_id: int

class Deployment(DeploymentBase):
    id: int
    status: str
    endpoint_url: Optional[str] = None
    created_at: datetime
    updated_at: datetime
    model_id: int
    
    class Config:
        from_attributes = True

# Evaluation schemas
class EvaluationBase(BaseModel):
    name: str
    test_data_path: str

class EvaluationCreate(EvaluationBase):
    model_id: int

class Evaluation(EvaluationBase):
    id: int
    status: str
    metrics: Optional[Dict[str, Any]] = None
    created_at: datetime
    model_id: int
    
    class Config:
        from_attributes = True

# Training Job schemas
class TrainingJobBase(BaseModel):
    name: str
    job_type: str
    config: Optional[Dict[str, Any]] = None

class TrainingJobCreate(TrainingJobBase):
    model_id: int

class TrainingJob(TrainingJobBase):
    id: int
    status: str
    logs: Optional[str] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    model_id: int
    
    class Config:
        from_attributes = True

# Metrics schemas
class DeploymentMetricsBase(BaseModel):
    cpu_usage: Optional[float] = None
    memory_usage: Optional[float] = None
    gpu_usage: Optional[float] = None
    request_count: Optional[int] = None
    response_time: Optional[float] = None
    error_rate: Optional[float] = None

class DeploymentMetrics(DeploymentMetricsBase):
    id: int
    timestamp: datetime
    deployment_id: int
    
    class Config:
        from_attributes = True

# Dashboard schemas
class ModelStatusSummary(BaseModel):
    total_models: int
    training_models: int
    deployed_models: int
    error_models: int

class PerformanceMetrics(BaseModel):
    model_id: int
    model_name: str
    accuracy: Optional[float] = None
    precision: Optional[float] = None
    recall: Optional[float] = None
    f1_score: Optional[float] = None
    inference_time: Optional[float] = None
    throughput: Optional[float] = None
    last_updated: datetime

