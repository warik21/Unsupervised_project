from typing import List, Any

from pydantic import BaseModel, validator
from sklearn.metrics import precision_recall_curve
import matplotlib.pyplot as plt

from sklearn.metrics import average_precision_score


class TrainingMetrics(BaseModel):
    train_losses: List[float]
    train_accuracies: List[float]
    val_losses: List[float]
    val_accuracies: List[float]
    predictions: Any


# precision, recall, _ = precision_recall_curve(y_true, y_score)
# prauc_score = average_precision_score(y_true, y_score, average='macro')

# plt.plot(recall, precision, marker='.', label='PRAUC = %0.3f' % pra_score)
# plt.xlabel('Recall')
# plt.ylabel('Precision')
# plt.title('Precision-Recall Curve')
# plt.legend()
# plt.show()

