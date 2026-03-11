from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer
import uvicorn
import os
from dotenv import load_dotenv

from app.api.v1.api import api_router
from app.core.config import settings
from app.db.database import engine
from app.db import models

# Load environment variables
load_dotenv()

# Create database tables
models.Base.metadata.create_all(bind=engine)

app = FastAPI(
    title="AI Platform API",
    description="A comprehensive platform for training, deploying, and monitoring AI models",
    version="1.0.0",
    openapi_url=f"{settings.API_V1_STR}/openapi.json"
)

# Set up CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # React app URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API router
app.include_router(api_router, prefix=settings.API_V1_STR)

@app.get("/")
async def root():
    return {"message": "AI Platform API is running"}

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )

