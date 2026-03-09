# UFC Pre-Fight Win Predictor (MSIS 522 HW1)

## Project Overview
This repository implements the complete data science workflow on UFC fights (1993-2025), from data engineering and descriptive analytics to predictive modeling, explainability, and deployment.

## Explicit Prediction Task
- **Task type:** Binary classification.
- **Target:** `y_red_win = 1` if `fight_winner == "RED"`, else `0` for `fight_winner == "BLUE"`.
- **Business framing:** Pre-fight win probability estimator for scouting, matchup analysis, and decision support.
- **Output:** Predicted class (`RED wins`/`BLUE wins`) and predicted probability `p(RED wins)`.

## Leakage Policy (Pre-Fight Only)
- Only features known before fight start are used.
- Keeps only completed fights with winner in {RED, BLUE}.
- Drops post-fight/in-fight variables (winner, outcome, end round/clock, judges, scores, etc.).
- Excludes **all** variables from `fighter_stats.csv`.
- Uses strict leakage guardrails in code (`validate_no_leakage`).

## Repository Structure
```
.
+-- app.py
+-- requirements.txt
+-- README.md
+-- runtime.txt
+-- .gitignore
+-- Makefile
+-- data/
¦   +-- raw/
¦   +-- processed/
+-- models/
+-- outputs/
¦   +-- metrics.json
¦   +-- best_params.json
¦   +-- model_comparison.csv
¦   +-- nn_tuning_results.csv
¦   +-- figures/
+-- src/
```

## Models Implemented
- Logistic Regression (baseline)
- Decision Tree (GridSearchCV, 5-fold StratifiedKFold)
- Random Forest (GridSearchCV, 5-fold StratifiedKFold)
- XGBoost (GridSearchCV, 5-fold StratifiedKFold)
- Neural Net MLP (TensorFlow/Keras, 2 hidden layers, ReLU)

All evaluated on test set with Accuracy, Precision, Recall, F1, ROC-AUC.

## Bonus (+1)
Neural network hyperparameter tuning grid is included:
- `hidden_units`: [64, 128]
- `dropout`: [0.0, 0.2]
- `learning_rate`: [1e-3, 5e-4]

Artifacts:
- `outputs/nn_tuning_results.csv`
- `outputs/figures/nn_tuning_heatmap.png`

## Setup
```bash
pip install -r requirements.txt
```

## Run Full Workflow
```bash
make all
```

Equivalent module commands:
```bash
python -m src.make_dataset
python -m src.eda
python -m src.train_models
python -m src.explain_shap
```

## Run Streamlit Locally
```bash
make app
# or
streamlit run app.py
```

## Streamlit App Tabs
1. Executive Summary
2. Descriptive Analytics
3. Model Performance
4. Explainability & Interactive Prediction

Tab 4 includes real-time prediction from saved models and SHAP waterfall explanation for user-entered features.

## Streamlit Community Cloud Deployment
1. Push this repository to GitHub.
2. Go to Streamlit Community Cloud and connect your GitHub account.
3. Click **New app** and select this repo/branch.
4. Set main file path to `app.py`.
5. Deploy.

## Submission Checklist
- [ ] GitHub repo link
- [ ] Public Streamlit app link
- [ ] `make all` runs and generates required artifacts
- [ ] Interactive prediction + probability + SHAP waterfall works in deployed app