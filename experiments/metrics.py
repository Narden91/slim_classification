
import torch
from slim_gsgp.classification.utils import apply_sigmoid

def get_accuracy(y_true, y_pred):
    """
    Calculate accuracy.
    y_true: true labels (0/1)
    y_pred: predicted labels (0/1)
    """
    correct = (y_true == y_pred).float()
    return torch.mean(correct)

def get_confusion_matrix_elements(y_true, y_pred):
    """
    Return TP, TN, FP, FN.
    """
    TP = ((y_pred == 1) & (y_true == 1)).sum().float()
    TN = ((y_pred == 0) & (y_true == 0)).sum().float()
    FP = ((y_pred == 1) & (y_true == 0)).sum().float()
    FN = ((y_pred == 0) & (y_true == 1)).sum().float()
    return TP, TN, FP, FN

def get_precision_recall_f1(y_true, y_pred):
    TP, TN, FP, FN = get_confusion_matrix_elements(y_true, y_pred)
    
    precision = TP / (TP + FP) if (TP + FP) > 0 else torch.tensor(0.0)
    recall = TP / (TP + FN) if (TP + FN) > 0 else torch.tensor(0.0)
    
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else torch.tensor(0.0)
    
    return precision, recall, f1

def get_specificity(y_true, y_pred):
    """
    Calculate specificity (True Negative Rate).
    Specificity = TN / (TN + FP)
    """
    TP, TN, FP, FN = get_confusion_matrix_elements(y_true, y_pred)
    result = TN / (TN + FP) if (TN + FP) > 0 else torch.tensor(0.0)
    return result

def get_all_metrics(y_true, y_pred_continuous, sigmoid_scale=1.0):
    """
    Calculate all classification metrics.
    y_true: tensor of true labels (0/1)
    y_pred_continuous: tensor of continuous predictions (logits or raw scores)
    sigmoid_scale: scaling factor applied before sigmoid
    """
    y_pred_probs = apply_sigmoid(y_pred_continuous, scaling_factor=sigmoid_scale, _skip_validation=True)
    y_pred_labels = (y_pred_probs >= 0.5).float()
    
    acc = get_accuracy(y_true, y_pred_labels)
    prec, rec, f1 = get_precision_recall_f1(y_true, y_pred_labels)
    spec = get_specificity(y_true, y_pred_labels)
    
    return {
        "main_metric_name": "accuracy",
        "main_metric_value": acc.item(),
        "accuracy": acc.item(),
        "precision": prec.item(),
        "recall": rec.item(),
        "specificity": spec.item(),
        "f1_score": f1.item()
    }
