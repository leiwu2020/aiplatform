from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form
from sqlalchemy.orm import Session
from typing import List
import os
import aiofiles
from datetime import datetime

from app.db.database import get_db
from app.db import models
from app.schemas.models import Dataset, DatasetCreate
from app.core.config import settings

router = APIRouter()

@router.post("/upload", response_model=Dataset)
async def upload_dataset(
    file: UploadFile = File(...),
    name: str = Form(...),
    description: str = Form(""),
    db: Session = Depends(get_db)
):
    """Upload a training dataset"""
    
    # Create data directory if it doesn't exist
    os.makedirs(settings.DATA_STORAGE_PATH, exist_ok=True)
    
    # Generate unique filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{timestamp}_{file.filename}"
    file_path = os.path.join(settings.DATA_STORAGE_PATH, filename)
    
    # Save file
    async with aiofiles.open(file_path, 'wb') as f:
        content = await file.read()
        await f.write(content)
    
    # Create dataset record
    db_dataset = models.Dataset(
        name=name,
        description=description,
        file_path=file_path,
        file_size=len(content),
        file_type=file.content_type or "unknown",
        status="uploaded",
        metadata={"original_filename": file.filename},
        owner_id=1  # TODO: Get from authentication
    )
    
    db.add(db_dataset)
    db.commit()
    db.refresh(db_dataset)
    
    return db_dataset

@router.get("/", response_model=List[Dataset])
def get_datasets(
    skip: int = 0,
    limit: int = 100,
    db: Session = Depends(get_db)
):
    """Get all datasets"""
    datasets = db.query(models.Dataset).offset(skip).limit(limit).all()
    return datasets

@router.get("/{dataset_id}", response_model=Dataset)
def get_dataset(
    dataset_id: int,
    db: Session = Depends(get_db)
):
    """Get a specific dataset"""
    dataset = db.query(models.Dataset).filter(models.Dataset.id == dataset_id).first()
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")
    return dataset

@router.delete("/{dataset_id}")
def delete_dataset(
    dataset_id: int,
    db: Session = Depends(get_db)
):
    """Delete a dataset"""
    dataset = db.query(models.Dataset).filter(models.Dataset.id == dataset_id).first()
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")
    
    # Delete file if it exists
    if os.path.exists(dataset.file_path):
        os.remove(dataset.file_path)
    
    db.delete(dataset)
    db.commit()
    
    return {"message": "Dataset deleted successfully"}

