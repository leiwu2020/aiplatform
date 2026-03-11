from pydantic_settings import BaseSettings
from typing import Optional
import os

class Settings(BaseSettings):
    # API Configuration
    API_V1_STR: str = "/api/v1"
    SECRET_KEY: str = os.getenv("SECRET_KEY", "your-secret-key-here")
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    
    # Database
    DATABASE_URL: str = os.getenv("DATABASE_URL", "sqlite:///./aiplatform.db")
    REDIS_URL: str = os.getenv("REDIS_URL", "redis://localhost:6379")
    
    # ML Configuration
    MLFLOW_TRACKING_URI: str = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
    WANDB_API_KEY: Optional[str] = os.getenv("WANDB_API_KEY")
    HUGGINGFACE_HUB_TOKEN: Optional[str] = os.getenv("HUGGINGFACE_HUB_TOKEN")
    
    # Storage
    MODEL_STORAGE_PATH: str = os.getenv("MODEL_STORAGE_PATH", "./models")
    DATA_STORAGE_PATH: str = os.getenv("DATA_STORAGE_PATH", "./data")
    
    # Monitoring
    PROMETHEUS_PORT: int = int(os.getenv("PROMETHEUS_PORT", "9090"))
    GRAFANA_PORT: int = int(os.getenv("GRAFANA_PORT", "3000"))
    
    # Kubernetes
    KUBECONFIG_PATH: str = os.getenv("KUBECONFIG_PATH", "~/.kube/config")
    NAMESPACE: str = os.getenv("NAMESPACE", "aiplatform")
    
    class Config:
        env_file = ".env"

settings = Settings()

