from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from sqlalchemy.orm import Session
from typing import List

from app.db.database import get_db
from app.db import models
from app.schemas.models import Evaluation, EvaluationCreate
from app.services.evaluation_service import EvaluationService

router = APIRouter()

@router.post("/", response_model=Evaluation)
def create_evaluation(
    evaluation: EvaluationCreate,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db)
):
    """Create a new evaluation"""
    
    # Check if model exists
    model = db.query(models.Model).filter(models.Model.id == evaluation.model_id).first()
    if not model:
        raise HTTPException(status_code=404, detail="Model not found")
    
    # Create evaluation record
    db_evaluation = models.Evaluation(
        name=evaluation.name,
        test_data_path=evaluation.test_data_path,
        model_id=evaluation.model_id,
        status="pending"
    )
    
    db.add(db_evaluation)
    db.commit()
    db.refresh(db_evaluation)
    
    # Start evaluation in background
    background_tasks.add_task(run_evaluation, db_evaluation.id)
    
    return db_evaluation

@router.get("/", response_model=List[Evaluation])
def get_evaluations(
    skip: int = 0,
    limit: int = 100,
    db: Session = Depends(get_db)
):
    """Get all evaluations"""
    evaluations = db.query(models.Evaluation).offset(skip).limit(limit).all()
    return evaluations

@router.get("/{evaluation_id}", response_model=Evaluation)
def get_evaluation(
    evaluation_id: int,
    db: Session = Depends(get_db)
):
    """Get a specific evaluation"""
    evaluation = db.query(models.Evaluation).filter(models.Evaluation.id == evaluation_id).first()
    if not evaluation:
        raise HTTPException(status_code=404, detail="Evaluation not found")
    return evaluation

@router.get("/model/{model_id}", response_model=List[Evaluation])
def get_model_evaluations(
    model_id: int,
    db: Session = Depends(get_db)
):
    """Get all evaluations for a specific model"""
    evaluations = db.query(models.Evaluation).filter(models.Evaluation.model_id == model_id).all()
    return evaluations

async def run_evaluation(evaluation_id: int):
    """Background task to run evaluation"""
    from app.db.database import SessionLocal
    from datetime import datetime
    
    db = SessionLocal()
    try:
        evaluation = db.query(models.Evaluation).filter(models.Evaluation.id == evaluation_id).first()
        if not evaluation:
            return
        
        # Update evaluation status
        evaluation.status = "running"
        db.commit()
        
        # Run evaluation
        evaluation_service = EvaluationService()
        result = await evaluation_service.evaluate_model(evaluation)
        
        if result["success"]:
            evaluation.status = "completed"
            evaluation.metrics = result["metrics"]
        else:
            evaluation.status = "error"
        
        db.commit()
        
    except Exception as e:
        evaluation = db.query(models.Evaluation).filter(models.Evaluation.id == evaluation_id).first()
        if evaluation:
            evaluation.status = "error"
            db.commit()
    finally:
        db.close()

