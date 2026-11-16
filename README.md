# SuperKart Sales Forecast - MLOps Pipeline
GitHub Actions pipeline: validate data -> prepare train/test (workflow artifact) -> tune Random Forest and Gradient Boosting with MLflow -> commit best model -> Streamlit app.

```
data/SuperKart.csv
scripts/register_data.py, data_prep.py, train.py
models/best_model.joblib, metrics.json
app.py, requirements.txt
.github/workflows/pipeline.yml
SuperKart_MLOps_Notebook.ipynb
```
