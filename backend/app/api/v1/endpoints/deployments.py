from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from sqlalchemy.orm import Session
from typing import List

from app.db.database import get_db
from app.db import models
from app.schemas.models import Deployment, DeploymentCreate
from app.services.deployment_service import DeploymentService

router = APIRouter()

@router.post("/", response_model=Deployment)
def create_deployment(
    deployment: DeploymentCreate,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db)
):
    """Create a new deployment"""
    
    # Check if model exists and is trained
    model = db.query(models.Model).filter(models.Model.id == deployment.model_id).first()
    if not model:
        raise HTTPException(status_code=404, detail="Model not found")
    
    if model.status != "trained":
        raise HTTPException(status_code=400, detail="Model must be trained before deployment")
    
    # Create deployment record
    db_deployment = models.Deployment(
        name=deployment.name,
        replicas=deployment.replicas,
        cpu_limit=deployment.cpu_limit,
        memory_limit=deployment.memory_limit,
        gpu_limit=deployment.gpu_limit,
        model_id=deployment.model_id,
        status="pending"
    )
    
    db.add(db_deployment)
    db.commit()
    db.refresh(db_deployment)
    
    # Start deployment in background
    background_tasks.add_task(deploy_model, db_deployment.id)
    
    return db_deployment

@router.get("/", response_model=List[Deployment])
def get_deployments(
    skip: int = 0,
    limit: int = 100,
    db: Session = Depends(get_db)
):
    """Get all deployments"""
    deployments = db.query(models.Deployment).offset(skip).limit(limit).all()
    return deployments

@router.get("/{deployment_id}", response_model=Deployment)
def get_deployment(
    deployment_id: int,
    db: Session = Depends(get_db)
):
    """Get a specific deployment"""
    deployment = db.query(models.Deployment).filter(models.Deployment.id == deployment_id).first()
    if not deployment:
        raise HTTPException(status_code=404, detail="Deployment not found")
    return deployment

@router.post("/{deployment_id}/scale")
def scale_deployment(
    deployment_id: int,
    replicas: int,
    db: Session = Depends(get_db)
):
    """Scale a deployment"""
    deployment = db.query(models.Deployment).filter(models.Deployment.id == deployment_id).first()
    if not deployment:
        raise HTTPException(status_code=404, detail="Deployment not found")
    
    if deployment.status != "active":
        raise HTTPException(status_code=400, detail="Deployment must be active to scale")
    
    deployment.replicas = replicas
    db.commit()
    
    # TODO: Update Kubernetes deployment
    deployment_service = DeploymentService()
    deployment_service.scale_deployment(deployment_id, replicas)
    
    return {"message": f"Deployment scaled to {replicas} replicas"}

@router.delete("/{deployment_id}")
def delete_deployment(
    deployment_id: int,
    db: Session = Depends(get_db)
):
    """Delete a deployment"""
    deployment = db.query(models.Deployment).filter(models.Deployment.id == deployment_id).first()
    if not deployment:
        raise HTTPException(status_code=404, detail="Deployment not found")
    
    # TODO: Delete from Kubernetes
    deployment_service = DeploymentService()
    deployment_service.delete_deployment(deployment_id)
    
    db.delete(deployment)
    db.commit()
    
    return {"message": "Deployment deleted successfully"}

async def deploy_model(deployment_id: int):
    """Background task to deploy model"""
    from app.db.database import SessionLocal
    from datetime import datetime
    
    db = SessionLocal()
    try:
        deployment = db.query(models.Deployment).filter(models.Deployment.id == deployment_id).first()
        if not deployment:
            return
        
        # Update deployment status
        deployment.status = "deploying"
        db.commit()
        
        # Deploy model
        deployment_service = DeploymentService()
        result = await deployment_service.deploy_model(deployment)
        
        if result["success"]:
            deployment.status = "active"
            deployment.endpoint_url = result["endpoint_url"]
        else:
            deployment.status = "error"
        
        db.commit()
        
    except Exception as e:
        deployment = db.query(models.Deployment).filter(models.Deployment.id == deployment_id).first()
        if deployment:
            deployment.status = "error"
            db.commit()
    finally:
        db.close()

