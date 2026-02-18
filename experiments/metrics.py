
import torch

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

def get_all_metrics(y_true, y_pred_continuous):
    """
    Calculate all classification metrics.
    y_true: tensor of true labels (0/1)
    y_pred_continuous: tensor of continuous predictions (logits or raw scores)
    """
    # Threshold at 0 for signed errors/logits, or 0.5 for probabilities? 
    # SLIM often returns raw values or sigmoid. Check implementation.
    # In binary.py of slim, it uses sigmoid. 
    # Let's assume input needs to be thresholded.
    # If the range is large, 0 is a safer threshold for raw, but 0.5 for sigmoid.
    # Let's assume sigmoid output [0,1].
    
    y_pred_probs = torch.sigmoid(y_pred_continuous)
    y_pred_labels = (y_pred_probs >= 0.5).float()
    
    acc = get_accuracy(y_true, y_pred_labels)
    prec, rec, f1 = get_precision_recall_f1(y_true, y_pred_labels)
    spec = get_specificity(y_true, y_pred_labels)
    
    return {
        "accuracy": acc.item(),
        "precision": prec.item(),
        "recall": rec.item(),
        "specificity": spec.item(),
        "f1_score": f1.item()
    }
