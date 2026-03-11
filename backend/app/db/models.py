from sqlalchemy import Column, Integer, String, DateTime, Float, Boolean, Text, ForeignKey, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from datetime import datetime

Base = declarative_base()

class User(Base):
    __tablename__ = "users"
    
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True)
    email = Column(String, unique=True, index=True)
    hashed_password = Column(String)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    models = relationship("Model", back_populates="owner")
    datasets = relationship("Dataset", back_populates="owner")

class Dataset(Base):
    __tablename__ = "datasets"
    
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, index=True)
    description = Column(Text)
    file_path = Column(String)
    file_size = Column(Integer)
    file_type = Column(String)
    status = Column(String, default="uploaded")  # uploaded, processing, ready, error
    metadata = Column(JSON)
    created_at = Column(DateTime, default=datetime.utcnow)
    owner_id = Column(Integer, ForeignKey("users.id"))
    
    # Relationships
    owner = relationship("User", back_populates="datasets")
    models = relationship("Model", back_populates="dataset")

class Model(Base):
    __tablename__ = "models"
    
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, index=True)
    description = Column(Text)
    model_type = Column(String)  # custom, pretrain, finetune
    base_model = Column(String)  # For foundation models
    status = Column(String, default="created")  # created, training, trained, deployed, error
    model_path = Column(String)
    config = Column(JSON)
    metrics = Column(JSON)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    owner_id = Column(Integer, ForeignKey("users.id"))
    dataset_id = Column(Integer, ForeignKey("datasets.id"))
    
    # Relationships
    owner = relationship("User", back_populates="models")
    dataset = relationship("Dataset", back_populates="models")
    deployments = relationship("Deployment", back_populates="model")
    evaluations = relationship("Evaluation", back_populates="model")

class Deployment(Base):
    __tablename__ = "deployments"
    
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, index=True)
    status = Column(String, default="pending")  # pending, deploying, active, inactive, error
    endpoint_url = Column(String)
    replicas = Column(Integer, default=1)
    cpu_limit = Column(String)
    memory_limit = Column(String)
    gpu_limit = Column(String)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    model_id = Column(Integer, ForeignKey("models.id"))
    
    # Relationships
    model = relationship("Model", back_populates="deployments")
    metrics = relationship("DeploymentMetrics", back_populates="deployment")

class Evaluation(Base):
    __tablename__ = "evaluations"
    
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, index=True)
    status = Column(String, default="pending")  # pending, running, completed, error
    metrics = Column(JSON)
    test_data_path = Column(String)
    created_at = Column(DateTime, default=datetime.utcnow)
    model_id = Column(Integer, ForeignKey("models.id"))
    
    # Relationships
    model = relationship("Model", back_populates="evaluations")

class DeploymentMetrics(Base):
    __tablename__ = "deployment_metrics"
    
    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime, default=datetime.utcnow)
    cpu_usage = Column(Float)
    memory_usage = Column(Float)
    gpu_usage = Column(Float)
    request_count = Column(Integer)
    response_time = Column(Float)
    error_rate = Column(Float)
    deployment_id = Column(Integer, ForeignKey("deployments.id"))
    
    # Relationships
    deployment = relationship("Deployment", back_populates="metrics")

class TrainingJob(Base):
    __tablename__ = "training_jobs"
    
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, index=True)
    status = Column(String, default="pending")  # pending, running, completed, failed
    job_type = Column(String)  # custom_training, pretrain, finetune
    config = Column(JSON)
    logs = Column(Text)
    started_at = Column(DateTime)
    completed_at = Column(DateTime)
    model_id = Column(Integer, ForeignKey("models.id"))
    
    # Relationships
    model = relationship("Model")

