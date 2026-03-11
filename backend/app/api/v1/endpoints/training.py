from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from sqlalchemy.orm import Session
from typing import List
import asyncio
import json
from datetime import datetime

from app.db.database import get_db
from app.db import models
from app.schemas.models import TrainingJob, TrainingJobCreate
from app.services.training_service import TrainingService

router = APIRouter()

@router.post("/start", response_model=TrainingJob)
def start_training(
    training_job: TrainingJobCreate,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db)
):
    """Start a training job"""
    
    # Create training job record
    db_job = models.TrainingJob(
        name=training_job.name,
        job_type=training_job.job_type,
        config=training_job.config,
        model_id=training_job.model_id,
        status="pending"
    )
    
    db.add(db_job)
    db.commit()
    db.refresh(db_job)
    
    # Start training in background
    background_tasks.add_task(run_training, db_job.id, training_job.config)
    
    return db_job

@router.get("/jobs", response_model=List[TrainingJob])
def get_training_jobs(
    skip: int = 0,
    limit: int = 100,
    db: Session = Depends(get_db)
):
    """Get all training jobs"""
    jobs = db.query(models.TrainingJob).offset(skip).limit(limit).all()
    return jobs

@router.get("/jobs/{job_id}", response_model=TrainingJob)
def get_training_job(
    job_id: int,
    db: Session = Depends(get_db)
):
    """Get a specific training job"""
    job = db.query(models.TrainingJob).filter(models.TrainingJob.id == job_id).first()
    if not job:
        raise HTTPException(status_code=404, detail="Training job not found")
    return job

@router.post("/stop/{job_id}")
def stop_training(
    job_id: int,
    db: Session = Depends(get_db)
):
    """Stop a training job"""
    job = db.query(models.TrainingJob).filter(models.TrainingJob.id == job_id).first()
    if not job:
        raise HTTPException(status_code=404, detail="Training job not found")
    
    if job.status == "running":
        job.status = "stopped"
        db.commit()
        return {"message": "Training job stopped"}
    else:
        raise HTTPException(status_code=400, detail="Training job is not running")

async def run_training(job_id: int, config: dict):
    """Background task to run training"""
    from app.db.database import SessionLocal
    
    db = SessionLocal()
    try:
        job = db.query(models.TrainingJob).filter(models.TrainingJob.id == job_id).first()
        if not job:
            return
        
        # Update job status
        job.status = "running"
        job.started_at = datetime.utcnow()
        db.commit()
        
        # Initialize training service
        training_service = TrainingService()
        
        # Run training based on job type
        if job.job_type == "custom_training":
            result = await training_service.train_custom_model(job.model_id, config)
        elif job.job_type == "pretrain":
            result = await training_service.pretrain_foundation_model(job.model_id, config)
        elif job.job_type == "finetune":
            result = await training_service.finetune_foundation_model(job.model_id, config)
        else:
            raise ValueError(f"Unknown job type: {job.job_type}")
        
        # Update job with results
        job.status = "completed" if result["success"] else "failed"
        job.completed_at = datetime.utcnow()
        job.logs = result.get("logs", "")
        db.commit()
        
        # Update model status
        model = db.query(models.Model).filter(models.Model.id == job.model_id).first()
        if model:
            model.status = "trained" if result["success"] else "error"
            model.model_path = result.get("model_path")
            model.metrics = result.get("metrics")
            db.commit()
            
    except Exception as e:
        # Update job status on error
        job = db.query(models.TrainingJob).filter(models.TrainingJob.id == job_id).first()
        if job:
            job.status = "failed"
            job.completed_at = datetime.utcnow()
            job.logs = str(e)
            db.commit()
    finally:
        db.close()
