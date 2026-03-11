import os
import json
import docker
import yaml
from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)

class DeploymentService:
    def __init__(self):
        self.docker_client = docker.from_env()
        self.namespace = os.getenv("NAMESPACE", "aiplatform")

    async def deploy_model(self, deployment) -> Dict[str, Any]:
        """Deploy a model to Kubernetes"""
        try:
            logger.info(f"Deploying model {deployment.model_id}")
            
            # Create Docker image
            image_name = f"aiplatform-model-{deployment.id}"
            image_tag = await self._build_docker_image(deployment, image_name)
            
            # Create Kubernetes deployment
            k8s_manifest = self._create_k8s_manifest(deployment, image_tag)
            
            # Apply Kubernetes manifest
            success = await self._apply_k8s_manifest(k8s_manifest)
            
            if success:
                endpoint_url = f"http://model-{deployment.id}.{self.namespace}.svc.cluster.local:8000"
                return {
                    "success": True,
                    "endpoint_url": endpoint_url,
                    "message": "Model deployed successfully"
                }
            else:
                return {
                    "success": False,
                    "error": "Failed to deploy to Kubernetes"
                }
                
        except Exception as e:
            logger.error(f"Error deploying model: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }

    async def scale_deployment(self, deployment_id: int, replicas: int) -> bool:
        """Scale a deployment"""
        try:
            # This would scale the Kubernetes deployment
            logger.info(f"Scaling deployment {deployment_id} to {replicas} replicas")
            return True
        except Exception as e:
            logger.error(f"Error scaling deployment: {str(e)}")
            return False

    async def delete_deployment(self, deployment_id: int) -> bool:
        """Delete a deployment"""
        try:
            # This would delete the Kubernetes deployment
            logger.info(f"Deleting deployment {deployment_id}")
            return True
        except Exception as e:
            logger.error(f"Error deleting deployment: {str(e)}")
            return False

    async def _build_docker_image(self, deployment, image_name: str) -> str:
        """Build Docker image for the model"""
        try:
            # Create Dockerfile
            dockerfile_content = self._create_dockerfile(deployment)
            
            # Create build context
            build_context = self._create_build_context(deployment, dockerfile_content)
            
            # Build image
            image_tag = f"{image_name}:latest"
            image, build_logs = self.docker_client.images.build(
                path=build_context,
                tag=image_tag,
                rm=True
            )
            
            logger.info(f"Built Docker image: {image_tag}")
            return image_tag
            
        except Exception as e:
            logger.error(f"Error building Docker image: {str(e)}")
            raise

    def _create_dockerfile(self, deployment) -> str:
        """Create Dockerfile for model deployment"""
        return f"""
FROM python:3.9-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy model files
COPY model/ ./model/

# Copy inference code
COPY inference.py .

# Expose port
EXPOSE 8000

# Run inference server
CMD ["python", "inference.py"]
"""

    def _create_build_context(self, deployment, dockerfile_content: str) -> str:
        """Create build context directory"""
        import tempfile
        import shutil
        
        # Create temporary directory
        build_dir = tempfile.mkdtemp()
        
        # Write Dockerfile
        with open(os.path.join(build_dir, "Dockerfile"), "w") as f:
            f.write(dockerfile_content)
        
        # Copy model files
        model_dir = os.path.join(build_dir, "model")
        os.makedirs(model_dir, exist_ok=True)
        
        if deployment.model.model_path:
            shutil.copytree(deployment.model.model_path, model_dir, dirs_exist_ok=True)
        
        # Create requirements.txt
        requirements = """
fastapi==0.104.1
uvicorn==0.24.0
torch==2.1.0
transformers==4.35.2
numpy==1.24.4
"""
        with open(os.path.join(build_dir, "requirements.txt"), "w") as f:
            f.write(requirements)
        
        # Create inference.py
        inference_code = self._create_inference_code()
        with open(os.path.join(build_dir, "inference.py"), "w") as f:
            f.write(inference_code)
        
        return build_dir

    def _create_inference_code(self) -> str:
        """Create inference server code"""
        return """
from fastapi import FastAPI
import torch
from transformers import AutoTokenizer, AutoModel
import json
import os

app = FastAPI()

# Load model and tokenizer
model_path = "./model"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModel.from_pretrained(model_path)
model.eval()

@app.post("/predict")
async def predict(data: dict):
    try:
        # Process input
        text = data.get("text", "")
        
        # Tokenize
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
        
        # Predict
        with torch.no_grad():
            outputs = model(**inputs)
            predictions = outputs.last_hidden_state.mean(dim=1)
        
        return {
            "predictions": predictions.tolist(),
            "status": "success"
        }
    except Exception as e:
        return {
            "error": str(e),
            "status": "error"
        }

@app.get("/health")
async def health():
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
"""

    def _create_k8s_manifest(self, deployment, image_tag: str) -> Dict[str, Any]:
        """Create Kubernetes deployment manifest"""
        return {
            "apiVersion": "apps/v1",
            "kind": "Deployment",
            "metadata": {
                "name": f"model-{deployment.id}",
                "namespace": self.namespace
            },
            "spec": {
                "replicas": deployment.replicas,
                "selector": {
                    "matchLabels": {
                        "app": f"model-{deployment.id}"
                    }
                },
                "template": {
                    "metadata": {
                        "labels": {
                            "app": f"model-{deployment.id}"
                        }
                    },
                    "spec": {
                        "containers": [
                            {
                                "name": f"model-{deployment.id}",
                                "image": image_tag,
                                "ports": [
                                    {
                                        "containerPort": 8000
                                    }
                                ],
                                "resources": {
                                    "limits": {
                                        "cpu": deployment.cpu_limit or "1",
                                        "memory": deployment.memory_limit or "1Gi"
                                    },
                                    "requests": {
                                        "cpu": "0.5",
                                        "memory": "512Mi"
                                    }
                                }
                            }
                        ]
                    }
                }
            }
        }

    async def _apply_k8s_manifest(self, manifest: Dict[str, Any]) -> bool:
        """Apply Kubernetes manifest"""
        try:
            # This would use kubectl or Kubernetes Python client
            # For now, return True as a placeholder
            logger.info("Applying Kubernetes manifest")
            return True
        except Exception as e:
            logger.error(f"Error applying Kubernetes manifest: {str(e)}")
            return False

