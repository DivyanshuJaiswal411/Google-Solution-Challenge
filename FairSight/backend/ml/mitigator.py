from fairlearn.postprocessing import ThresholdOptimizer
from backend.ml.analyzer import calculate_bias_score

def apply_mitigation(model_info):
    """
    Applies Fairlearn's ThresholdOptimizer to automatically adjust decision thresholds
    for different sensitive groups to achieve equality.
    """
    model = model_info["model"]
    X_train = model_info["X_train"]
    y_train = model_info["y_train"]
    A_train = model_info["A_train"]
    X_test = model_info["X_test"]
    y_test = model_info["y_test"]
    A_test = model_info["sensitive_features_test"]
    
    # We use ThresholdOptimizer to post-process the predictions
    optimizer = ThresholdOptimizer(
        estimator=model,
        constraints="demographic_parity",
        prefit=True
    )
    
    # Fit the optimizer
    optimizer.fit(X_train, y_train, sensitive_features=A_train)
    
    # Predict using mitigated thresholds
    mitigated_preds = optimizer.predict(X_test, sensitive_features=A_test)
    
    from sklearn.metrics import accuracy_score
    from fairlearn.metrics import demographic_parity_difference, equalized_odds_difference
    
    metrics = {
        "accuracy": round(accuracy_score(y_test, mitigated_preds), 4),
        "demographic_parity_difference": round(demographic_parity_difference(y_test, mitigated_preds, sensitive_features=A_test), 4),
        "equalized_odds_difference": round(equalized_odds_difference(y_test, mitigated_preds, sensitive_features=A_test), 4)
    }
    
    return metrics
