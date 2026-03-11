from fastapi import APIRouter
from app.api.v1.endpoints import datasets, models, deployments, evaluations, training, monitoring, dashboard

api_router = APIRouter()

api_router.include_router(datasets.router, prefix="/datasets", tags=["datasets"])
api_router.include_router(models.router, prefix="/models", tags=["models"])
api_router.include_router(deployments.router, prefix="/deployments", tags=["deployments"])
api_router.include_router(evaluations.router, prefix="/evaluations", tags=["evaluations"])
api_router.include_router(training.router, prefix="/training", tags=["training"])
api_router.include_router(monitoring.router, prefix="/monitoring", tags=["monitoring"])
api_router.include_router(dashboard.router, prefix="/dashboard", tags=["dashboard"])

