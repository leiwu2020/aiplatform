from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from sqlalchemy import func
from typing import List, Dict, Any

from app.db.database import get_db
from app.db import models
from app.schemas.models import ModelStatusSummary, PerformanceMetrics

router = APIRouter()

@router.get("/overview")
def get_dashboard_overview(db: Session = Depends(get_db)):
    """Get dashboard overview statistics"""
    
    # Model statistics
    total_models = db.query(models.Model).count()
    training_models = db.query(models.Model).filter(models.Model.status == "training").count()
    deployed_models = db.query(models.Model).filter(models.Model.status == "deployed").count()
    error_models = db.query(models.Model).filter(models.Model.status == "error").count()
    
    # Dataset statistics
    total_datasets = db.query(models.Dataset).count()
    ready_datasets = db.query(models.Dataset).filter(models.Dataset.status == "ready").count()
    
    # Deployment statistics
    total_deployments = db.query(models.Deployment).count()
    active_deployments = db.query(models.Deployment).filter(models.Deployment.status == "active").count()
    
    # Training job statistics
    total_jobs = db.query(models.TrainingJob).count()
    running_jobs = db.query(models.TrainingJob).filter(models.TrainingJob.status == "running").count()
    completed_jobs = db.query(models.TrainingJob).filter(models.TrainingJob.status == "completed").count()
    
    return {
        "models": {
            "total": total_models,
            "training": training_models,
            "deployed": deployed_models,
            "error": error_models
        },
        "datasets": {
            "total": total_datasets,
            "ready": ready_datasets
        },
        "deployments": {
            "total": total_deployments,
            "active": active_deployments
        },
        "training_jobs": {
            "total": total_jobs,
            "running": running_jobs,
            "completed": completed_jobs
        }
    }

@router.get("/recent-activity")
def get_recent_activity(db: Session = Depends(get_db)):
    """Get recent activity across the platform"""
    
    # Recent models
    recent_models = db.query(models.Model).order_by(models.Model.created_at.desc()).limit(5).all()
    
    # Recent deployments
    recent_deployments = db.query(models.Deployment).order_by(models.Deployment.created_at.desc()).limit(5).all()
    
    # Recent training jobs
    recent_jobs = db.query(models.TrainingJob).order_by(models.TrainingJob.started_at.desc()).limit(5).all()
    
    activities = []
    
    # Add model activities
    for model in recent_models:
        activities.append({
            "type": "model",
            "id": model.id,
            "name": model.name,
            "status": model.status,
            "timestamp": model.created_at,
            "action": "created"
        })
    
    # Add deployment activities
    for deployment in recent_deployments:
        activities.append({
            "type": "deployment",
            "id": deployment.id,
            "name": deployment.name,
            "status": deployment.status,
            "timestamp": deployment.created_at,
            "action": "deployed"
        })
    
    # Add training job activities
    for job in recent_jobs:
        activities.append({
            "type": "training",
            "id": job.id,
            "name": job.name,
            "status": job.status,
            "timestamp": job.started_at or job.created_at,
            "action": "started" if job.status == "running" else "completed"
        })
    
    # Sort by timestamp
    activities.sort(key=lambda x: x["timestamp"], reverse=True)
    
    return {"activities": activities[:10]}

@router.get("/performance-metrics")
def get_performance_metrics(db: Session = Depends(get_db)):
    """Get performance metrics for all models"""
    
    # Get all trained models with their evaluations
    models_with_metrics = db.query(models.Model).filter(
        models.Model.status.in_(["trained", "deployed"])
    ).all()
    
    performance_data = []
    
    for model in models_with_metrics:
        # Get latest evaluation
        latest_evaluation = db.query(models.Evaluation).filter(
            models.Evaluation.model_id == model.id,
            models.Evaluation.status == "completed"
        ).order_by(models.Evaluation.created_at.desc()).first()
        
        if latest_evaluation and latest_evaluation.metrics:
            performance_data.append({
                "model_id": model.id,
                "model_name": model.name,
                "model_type": model.model_type,
                "accuracy": latest_evaluation.metrics.get("accuracy"),
                "precision": latest_evaluation.metrics.get("precision"),
                "recall": latest_evaluation.metrics.get("recall"),
                "f1_score": latest_evaluation.metrics.get("f1_score"),
                "last_updated": latest_evaluation.created_at
            })
    
    return {"performance_metrics": performance_data}

@router.get("/resource-usage")
def get_resource_usage(db: Session = Depends(get_db)):
    """Get resource usage across all deployments"""
    
    # Get all active deployments
    active_deployments = db.query(models.Deployment).filter(
        models.Deployment.status == "active"
    ).all()
    
    resource_data = []
    
    for deployment in active_deployments:
        # Get latest metrics
        latest_metrics = db.query(models.DeploymentMetrics).filter(
            models.DeploymentMetrics.deployment_id == deployment.id
        ).order_by(models.DeploymentMetrics.timestamp.desc()).first()
        
        if latest_metrics:
            resource_data.append({
                "deployment_id": deployment.id,
                "deployment_name": deployment.name,
                "model_name": deployment.model.name,
                "cpu_usage": latest_metrics.cpu_usage,
                "memory_usage": latest_metrics.memory_usage,
                "gpu_usage": latest_metrics.gpu_usage,
                "request_count": latest_metrics.request_count,
                "response_time": latest_metrics.response_time,
                "error_rate": latest_metrics.error_rate,
                "last_updated": latest_metrics.timestamp
            })
    
    return {"resource_usage": resource_data}

