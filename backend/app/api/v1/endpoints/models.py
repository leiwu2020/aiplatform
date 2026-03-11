from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import List

from app.db.database import get_db
from app.db import models
from app.schemas.models import Model, ModelCreate, ModelUpdate

router = APIRouter()

@router.post("/", response_model=Model)
def create_model(
    model: ModelCreate,
    db: Session = Depends(get_db)
):
    """Create a new model"""
    db_model = models.Model(
        name=model.name,
        description=model.description,
        model_type=model.model_type,
        base_model=model.base_model,
        config=model.config,
        owner_id=1,  # TODO: Get from authentication
        dataset_id=model.dataset_id
    )
    
    db.add(db_model)
    db.commit()
    db.refresh(db_model)
    
    return db_model

@router.get("/", response_model=List[Model])
def get_models(
    skip: int = 0,
    limit: int = 100,
    db: Session = Depends(get_db)
):
    """Get all models"""
    models_list = db.query(models.Model).offset(skip).limit(limit).all()
    return models_list

@router.get("/{model_id}", response_model=Model)
def get_model(
    model_id: int,
    db: Session = Depends(get_db)
):
    """Get a specific model"""
    model = db.query(models.Model).filter(models.Model.id == model_id).first()
    if not model:
        raise HTTPException(status_code=404, detail="Model not found")
    return model

@router.put("/{model_id}", response_model=Model)
def update_model(
    model_id: int,
    model_update: ModelUpdate,
    db: Session = Depends(get_db)
):
    """Update a model"""
    model = db.query(models.Model).filter(models.Model.id == model_id).first()
    if not model:
        raise HTTPException(status_code=404, detail="Model not found")
    
    for field, value in model_update.dict(exclude_unset=True).items():
        setattr(model, field, value)
    
    db.commit()
    db.refresh(model)
    
    return model

@router.delete("/{model_id}")
def delete_model(
    model_id: int,
    db: Session = Depends(get_db)
):
    """Delete a model"""
    model = db.query(models.Model).filter(models.Model.id == model_id).first()
    if not model:
        raise HTTPException(status_code=404, detail="Model not found")
    
    db.delete(model)
    db.commit()
    
    return {"message": "Model deleted successfully"}

