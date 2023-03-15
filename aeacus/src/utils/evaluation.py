from typing import List, Any

from pydantic import BaseModel, validator

class TrainingMetrics(BaseModel):
    train_losses: List[float]
    train_accuracies: List[float]
    val_losses: List[float]
    val_accuracies: List[float]
    predictions: Any

class ModelEvaluators(BaseModel):
    accuracy: float
    precision: float
    recall: float
    f1: float
    def print_self(self):
        print('Accuracy: {} - Precision: {} - Recall: {} - F1: {}')