import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from fairlearn.metrics import demographic_parity_difference, equalized_odds_difference

def train_and_evaluate(df, target_col, sensitive_col):
    # Basic preprocessing for prototype
    # Drop categorical columns that aren't target or sensitive for simplicity in hackathon
    
    # Simple label encoding
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = df[col].astype('category').cat.codes
            
    y = df[target_col]
    X = df.drop(columns=[target_col])
    sensitive_features = df[sensitive_col]
    
    X_train, X_test, y_train, y_test, A_train, A_test = train_test_split(
        X, y, sensitive_features, test_size=0.3, random_state=42
    )
    
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    
    preds = model.predict(X_test)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, preds)
    dpd = demographic_parity_difference(y_test, preds, sensitive_features=A_test)
    eod = equalized_odds_difference(y_test, preds, sensitive_features=A_test)
    
    # We invert DPD and EOD so higher is better for the UI, or just send raw
    
    metrics = {
        "accuracy": round(accuracy, 4),
        "demographic_parity_difference": round(dpd, 4),
        "equalized_odds_difference": round(eod, 4)
    }
    
    model_info = {
        "model": model,
        "X_train": X_train,
        "y_train": y_train,
        "A_train": A_train,
        "X_test": X_test,
        "y_test": y_test,
        "sensitive_features_test": A_test,
        "sensitive_col_name": sensitive_col
    }
    
    # Intersectional Bias Analysis Matrix (Mock for Prototype)
    # Detects compounding bias if multple sensitive subgroups overlap.
    intersectional_risk = dpd * 1.5 if eod > 0.1 else dpd 
    metrics["intersectional_bias_risk"] = round(intersectional_risk, 4)
    
    return {"metrics": metrics, "model_info": model_info}

def calculate_bias_score(metrics):
    # A fake composite score out of 100
    # 0 difference is perfect.
    
    dpd_penalty = metrics.get("demographic_parity_difference", 0) * 100
    eod_penalty = metrics.get("equalized_odds_difference", 0) * 100
    intersectional_penalty = metrics.get("intersectional_bias_risk", 0) * 100
    
    # Give higher weight to intersectional compounded biases
    penalty = (dpd_penalty + eod_penalty + (intersectional_penalty * 0.5)) / 2.5
    
    score = max(0, min(100, 100 - penalty))
    return round(score)
