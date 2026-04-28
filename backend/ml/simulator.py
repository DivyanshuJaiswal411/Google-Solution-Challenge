import numpy as np
from ml.analyzer import calculate_bias_score
from sklearn.metrics import accuracy_score
from fairlearn.metrics import demographic_parity_difference, equalized_odds_difference

def run_simulation(model_info, threshold=0.5):
    """
    Simulates changing the decision threshold globally on the predictions probabilities.
    For more advanced what-if (different thresholds per group), the UI would send a dictionary.
    Here we shift the threshold.
    """
    model = model_info["model"]
    X_test = model_info["X_test"]
    y_test = model_info["y_test"]
    A_test = model_info["sensitive_features_test"]
    
    # Get probabilities for class 1
    probs = model.predict_proba(X_test)[:, 1]
    
    # Apply new threshold
    sim_preds = (probs >= threshold).astype(int)
    
    accuracy = accuracy_score(y_test, sim_preds)
    dpd = demographic_parity_difference(y_test, sim_preds, sensitive_features=A_test)
    eod = equalized_odds_difference(y_test, sim_preds, sensitive_features=A_test)
    
    metrics = {
        "accuracy": round(accuracy, 4),
        "demographic_parity_difference": round(dpd, 4),
        "equalized_odds_difference": round(eod, 4)
    }
    
    return {
        "metrics": metrics,
        "bias_score": calculate_bias_score(metrics)
    }
