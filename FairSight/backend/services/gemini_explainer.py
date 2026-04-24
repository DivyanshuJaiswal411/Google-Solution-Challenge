import os
from google import genai
from pydantic import BaseModel
from typing import Dict, Any

def generate_compliance_report(metrics: Dict[str, Any]) -> str:
    """
    Uses Gemini API to interpret the bias metrics and generate a GDPR-aligned compliance report.
    """
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        return "WARNING: GEMINI_API_KEY environment variable not set. Please set it to generate the report locally, or add it to Cloud Run secrets."
    
    # Initialize the new Google GenAI SDK client
    client = genai.Client(api_key=api_key)
    
    prompt = f"""
    You are an expert AI Ethicist and GDPR Compliance Auditor.
    I have evaluated a machine learning model, and I need you to generate a professional 'Bias Compliance & Certification Report'.
    
    Here are the metrics for the model:
    {metrics}
    
    Please provide a report with the following structure:
    1. Executive Summary
    2. Identified Risks (Analyze the fairness metrics - note if Disparate Impact < 0.8 is a concern)
    3. Model Behavior Summary
    4. Compliance with GDPR & AI Ethics Principles
    5. Actionable Fixes & Recommendations
    
    Make the report professional, clear, and ready to be presented to non-technical stakeholders (e.g., HR, Finance, Compliance officers).
    Use markdown format.
    """
    try:
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=prompt,
        )
        return response.text
    except Exception as e:
        return f"Error communicating with Gemini API: {str(e)}"
