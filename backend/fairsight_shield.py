import functools
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("FairSight-Shield")

def fairsight_shield(model_info, active=True):
    """
    Decorator for AI model prediction functions.
    Intercepts the inputs, checks for fairness violations in real-time.
    If 'active' is True, it automatically overwrites a localized biased prediction
    using the backend mitigator logic.
    """
    def decorator(predict_func):
        @functools.wraps(predict_func)
        def wrapper(user_features):
            # 1. Run the original potentially biased prediction
            original_prediction = predict_func(user_features)
            
            if not active:
                return original_prediction
            
            # Simulated real-time check
            # In a real system, we'd check if this person belongs to an underprivileged group
            # and if their probability was just under the threshold, we apply the mitigated threshold.
            
            sensitive_col = model_info.get("sensitive_col_name")
            if sensitive_col in user_features:
                logger.info(f"🛡️ FairSight Shield Intercepting prediction for user with sensitive attribute: {sensitive_col}")
                # Fetch mitigated model / threshold
                # For demo purposes, if original predicts 0 but their profile fits the mitigated bounds, flip to 1.
                
                # Mock: Assume the original model rejected them (0), but we know they belong
                # to a protected class that the mitigation engine boosted.
                if original_prediction == 0:
                    logger.warning("🛡️ FairSight Shield detected high risk of discriminatory rejection! Mitigating outcome...")
                    return 1 # Flipped to approved
                    
            return original_prediction
        return wrapper
    return decorator
