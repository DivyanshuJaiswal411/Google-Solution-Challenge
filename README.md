# FairSight: Bias Intelligence & Correction Platform

![FairSight](https://img.shields.io/badge/Google-Solution%20Challenge-blue)
![React](https://img.shields.io/badge/Frontend-React%20%7C%20Vite-61DAFB)
![FastAPI](https://img.shields.io/badge/Backend-FastAPI-009688)

## 🌟 The Problem
Computer programs now make life-changing decisions about who gets a job, a bank loan, or medical care. However, when these programs learn from flawed historical data, they amplify discrimination.

## 💡 Our Solution
**FairSight** is an elite **Bias Intelligence & Correction Platform**. It doesn't just detect bias on a dashboard; it proactively mitigates it, provides an interactive What-If Simulator, scores fairness out of 100, and uses **Google Gemini** to generate automated, plain-English GDPR/AI Ethics compliance reports.

## 🔥 "UNBEATABLE" Hackathon Features (V2 Core)
1. **FairSight Runtime Shield SDK (`@fairsight_shield`):** We move beyond passive dashboards. Our Python middleware SDK actively intercepts live traffic predicting on biased models and autonomously overrides discriminatory outcomes in real-time.
2. **Counterfactual "Flip-Test" API:** Our system can take an individual's rejected profile, computationally "flip" their sensitive attribute (e.g., changing Gender from Female to Male), and re-run the prediction. If it changes to Approved, we return undeniable proof of individual discrimination.
3. **Intersectional Fairness Matrix:** AI bias compounds. Our analyzer doesn't just evaluate single vectors (Race); it evaluates compounded vector intersections (e.g., Hispanic + Female) to catch hidden discrimination zones.
4. **Bias Certification Score (0-100):** Evaluates 'Disparate Impact' and 'Equal Opportunity Difference' into an intuitive "credit score for fairness".
5. **What-If Simulator & Mitigation Edge:** Interactive sliders simulate Accuracy vs Fairness tradeoffs, followed by a 1-click Fairlearn `ThresholdOptimizer` mitigation.
6. **Auto-Compliance Report:** Uses Google's Gemini API to translate complex machine learning data into a generated GDPR/AI Ethics compliance document.

## 🏛 Architecture
- **Frontend:** React (Vite) + Tailwind CSS (Premium Dark Mode Glassmorphism)
- **Backend:** FastAPI (Python)
- **ML Engine:** `scikit-learn` & `fairlearn`
- **Google Cloud Services:** Cloud Run (Deployments), Gemini API (Auto Reporting).

## 🚀 Setup & Execution

### 1. Backend (FastAPI)
```bash
cd backend
pip install -r requirements.txt
# Set your Gemini API Key for reporting
export GEMINI_API_KEY="your-api-key"
# Run API
uvicorn main:app --reload
```

### 2. Frontend (React)
*(Stitch UI Code Generation Link/Setup here)*
```bash
cd frontend
npm install
npm run dev
```

## ☁️ Google Cloud Deployment (Cloud Run)
```bash
# Build & Deploy Backend
gcloud run deploy fairsight-backend --source ./backend --allow-unauthenticated --set-env-vars="GEMINI_API_KEY=your_key"
```

## 🎥 Demo Video Highlights
- Start with showing a dataset uploading and plotting the *Bias Certification Score*.
- Next, highlight the **What-If Simulator**, adjusting a slider to show how the Bias Score jumps from "Red (50)" to "Green (85)".
- Finally, click **"Generate Compliance Report"** to show a beautiful, Gemini-generated GDPR audit document!

Built for the Google Solution Challenge.
