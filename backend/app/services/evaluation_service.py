import os
import json
import torch
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)

class EvaluationService:
    def __init__(self):
        self.model_storage_path = os.getenv("MODEL_STORAGE_PATH", "./models")

    async def evaluate_model(self, evaluation) -> Dict[str, Any]:
        """Evaluate a model on test data"""
        try:
            logger.info(f"Evaluating model {evaluation.model_id}")
            
            # Load model
            model = await self._load_model(evaluation.model_id)
            if not model:
                return {
                    "success": False,
                    "error": "Model not found"
                }
            
            # Load test data
            test_data = await self._load_test_data(evaluation.test_data_path)
            if not test_data:
                return {
                    "success": False,
                    "error": "Test data not found"
                }
            
            # Run evaluation
            metrics = await self._run_evaluation(model, test_data)
            
            return {
                "success": True,
                "metrics": metrics
            }
            
        except Exception as e:
            logger.error(f"Error evaluating model: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }

    async def _load_model(self, model_id: int):
        """Load model for evaluation"""
        try:
            model_path = os.path.join(self.model_storage_path, f"model_{model_id}")
            
            if not os.path.exists(model_path):
                return None
            
            # Load model based on type
            if os.path.exists(os.path.join(model_path, "config.json")):
                # Hugging Face model
                from transformers import AutoTokenizer, AutoModel
                tokenizer = AutoTokenizer.from_pretrained(model_path)
                model = AutoModel.from_pretrained(model_path)
                return {"model": model, "tokenizer": tokenizer, "type": "huggingface"}
            else:
                # Custom PyTorch model
                import torch
                model = torch.load(os.path.join(model_path, "model.pt"))
                return {"model": model, "type": "custom"}
                
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            return None

    async def _load_test_data(self, test_data_path: str):
        """Load test data"""
        try:
            import pandas as pd
            
            if not os.path.exists(test_data_path):
                return None
            
            if test_data_path.endswith('.csv'):
                return pd.read_csv(test_data_path)
            elif test_data_path.endswith('.json'):
                return pd.read_json(test_data_path)
            else:
                return None
                
        except Exception as e:
            logger.error(f"Error loading test data: {str(e)}")
            return None

    async def _run_evaluation(self, model_info, test_data):
        """Run evaluation on test data"""
        try:
            model = model_info["model"]
            model_type = model_info["type"]
            
            if model_type == "huggingface":
                return await self._evaluate_huggingface_model(model_info, test_data)
            else:
                return await self._evaluate_custom_model(model, test_data)
                
        except Exception as e:
            logger.error(f"Error running evaluation: {str(e)}")
            return {}

    async def _evaluate_huggingface_model(self, model_info, test_data):
        """Evaluate Hugging Face model"""
        try:
            model = model_info["model"]
            tokenizer = model_info["tokenizer"]
            
            # Prepare test data
            texts = test_data["text"].tolist()
            labels = test_data["label"].tolist() if "label" in test_data.columns else None
            
            # Tokenize texts
            inputs = tokenizer(texts, return_tensors="pt", truncation=True, padding=True)
            
            # Get predictions
            with torch.no_grad():
                outputs = model(**inputs)
                predictions = outputs.last_hidden_state.mean(dim=1)
                
                # For classification, we'd need to add a classification head
                # For now, return basic metrics
                if labels:
                    # Convert to numpy for sklearn metrics
                    pred_labels = predictions.argmax(dim=1).numpy()
                    true_labels = np.array(labels)
                    
                    metrics = {
                        "accuracy": float(accuracy_score(true_labels, pred_labels)),
                        "precision": float(precision_score(true_labels, pred_labels, average='weighted')),
                        "recall": float(recall_score(true_labels, pred_labels, average='weighted')),
                        "f1_score": float(f1_score(true_labels, pred_labels, average='weighted'))
                    }
                else:
                    metrics = {
                        "model_output_shape": list(predictions.shape),
                        "mean_output": float(predictions.mean().item()),
                        "std_output": float(predictions.std().item())
                    }
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error evaluating Hugging Face model: {str(e)}")
            return {}

    async def _evaluate_custom_model(self, model, test_data):
        """Evaluate custom PyTorch model"""
        try:
            # Prepare test data
            if "features" in test_data.columns:
                features = test_data["features"].tolist()
            else:
                # Use all numeric columns as features
                numeric_cols = test_data.select_dtypes(include=[np.number]).columns
                features = test_data[numeric_cols].values.tolist()
            
            labels = test_data["label"].tolist() if "label" in test_data.columns else None
            
            # Convert to tensors
            X = torch.tensor(features, dtype=torch.float32)
            
            # Get predictions
            model.eval()
            with torch.no_grad():
                predictions = model(X)
                
                if labels:
                    # For classification
                    pred_labels = predictions.argmax(dim=1).numpy()
                    true_labels = np.array(labels)
                    
                    metrics = {
                        "accuracy": float(accuracy_score(true_labels, pred_labels)),
                        "precision": float(precision_score(true_labels, pred_labels, average='weighted')),
                        "recall": float(recall_score(true_labels, pred_labels, average='weighted')),
                        "f1_score": float(f1_score(true_labels, pred_labels, average='weighted'))
                    }
                else:
                    # For regression
                    metrics = {
                        "mean_prediction": float(predictions.mean().item()),
                        "std_prediction": float(predictions.std().item()),
                        "min_prediction": float(predictions.min().item()),
                        "max_prediction": float(predictions.max().item())
                    }
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error evaluating custom model: {str(e)}")
            return {}

