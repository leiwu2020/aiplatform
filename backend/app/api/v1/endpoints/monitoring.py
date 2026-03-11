from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import List, Dict, Any
from datetime import datetime, timedelta

from app.db.database import get_db
from app.db import models
from app.schemas.models import DeploymentMetrics

router = APIRouter()

@router.get("/metrics/{deployment_id}")
def get_deployment_metrics(
    deployment_id: int,
    hours: int = 24,
    db: Session = Depends(get_db)
):
    """Get metrics for a specific deployment"""
    
    # Check if deployment exists
    deployment = db.query(models.Deployment).filter(models.Deployment.id == deployment_id).first()
    if not deployment:
        raise HTTPException(status_code=404, detail="Deployment not found")
    
    # Get metrics from the last N hours
    start_time = datetime.utcnow() - timedelta(hours=hours)
    metrics = db.query(models.DeploymentMetrics).filter(
        models.DeploymentMetrics.deployment_id == deployment_id,
        models.DeploymentMetrics.timestamp >= start_time
    ).order_by(models.DeploymentMetrics.timestamp).all()
    
    return {
        "deployment_id": deployment_id,
        "deployment_name": deployment.name,
        "metrics": [
            {
                "timestamp": metric.timestamp.isoformat(),
                "cpu_usage": metric.cpu_usage,
                "memory_usage": metric.memory_usage,
                "gpu_usage": metric.gpu_usage,
                "request_count": metric.request_count,
                "response_time": metric.response_time,
                "error_rate": metric.error_rate
            }
            for metric in metrics
        ]
    }

@router.post("/metrics/{deployment_id}")
def record_metrics(
    deployment_id: int,
    metrics_data: Dict[str, Any],
    db: Session = Depends(get_db)
):
    """Record new metrics for a deployment"""
    
    # Check if deployment exists
    deployment = db.query(models.Deployment).filter(models.Deployment.id == deployment_id).first()
    if not deployment:
        raise HTTPException(status_code=404, detail="Deployment not found")
    
    # Create metrics record
    db_metrics = models.DeploymentMetrics(
        deployment_id=deployment_id,
        cpu_usage=metrics_data.get("cpu_usage"),
        memory_usage=metrics_data.get("memory_usage"),
        gpu_usage=metrics_data.get("gpu_usage"),
        request_count=metrics_data.get("request_count"),
        response_time=metrics_data.get("response_time"),
        error_rate=metrics_data.get("error_rate")
    )
    
    db.add(db_metrics)
    db.commit()
    db.refresh(db_metrics)
    
    return {"message": "Metrics recorded successfully"}

@router.get("/alerts")
def get_alerts(db: Session = Depends(get_db)):
    """Get active alerts for all deployments"""
    
    # Get all active deployments
    deployments = db.query(models.Deployment).filter(models.Deployment.status == "active").all()
    
    alerts = []
    for deployment in deployments:
        # Get latest metrics
        latest_metrics = db.query(models.DeploymentMetrics).filter(
            models.DeploymentMetrics.deployment_id == deployment.id
        ).order_by(models.DeploymentMetrics.timestamp.desc()).first()
        
        if latest_metrics:
            # Check for alerts
            if latest_metrics.cpu_usage and latest_metrics.cpu_usage > 80:
                alerts.append({
                    "deployment_id": deployment.id,
                    "deployment_name": deployment.name,
                    "type": "high_cpu",
                    "message": f"High CPU usage: {latest_metrics.cpu_usage}%",
                    "timestamp": latest_metrics.timestamp
                })
            
            if latest_metrics.memory_usage and latest_metrics.memory_usage > 80:
                alerts.append({
                    "deployment_id": deployment.id,
                    "deployment_name": deployment.name,
                    "type": "high_memory",
                    "message": f"High memory usage: {latest_metrics.memory_usage}%",
                    "timestamp": latest_metrics.timestamp
                })
            
            if latest_metrics.error_rate and latest_metrics.error_rate > 5:
                alerts.append({
                    "deployment_id": deployment.id,
                    "deployment_name": deployment.name,
                    "type": "high_error_rate",
                    "message": f"High error rate: {latest_metrics.error_rate}%",
                    "timestamp": latest_metrics.timestamp
                })
    
    return {"alerts": alerts}

@router.get("/health/{deployment_id}")
def get_deployment_health(
    deployment_id: int,
    db: Session = Depends(get_db)
):
    """Get health status of a deployment"""
    
    deployment = db.query(models.Deployment).filter(models.Deployment.id == deployment_id).first()
    if not deployment:
        raise HTTPException(status_code=404, detail="Deployment not found")
    
    # Get latest metrics
    latest_metrics = db.query(models.DeploymentMetrics).filter(
        models.DeploymentMetrics.deployment_id == deployment_id
    ).order_by(models.DeploymentMetrics.timestamp.desc()).first()
    
    health_status = "healthy"
    issues = []
    
    if latest_metrics:
        if latest_metrics.cpu_usage and latest_metrics.cpu_usage > 80:
            health_status = "warning"
            issues.append("High CPU usage")
        
        if latest_metrics.memory_usage and latest_metrics.memory_usage > 80:
            health_status = "warning"
            issues.append("High memory usage")
        
        if latest_metrics.error_rate and latest_metrics.error_rate > 5:
            health_status = "critical"
            issues.append("High error rate")
    
    return {
        "deployment_id": deployment_id,
        "deployment_name": deployment.name,
        "status": deployment.status,
        "health_status": health_status,
        "issues": issues,
        "last_updated": latest_metrics.timestamp if latest_metrics else None
    }

