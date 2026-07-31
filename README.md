# Tourism Package Prediction — MLOps Pipeline

"Visit with Us" is introducing a new **Wellness Tourism Package**. This project builds an
end-to-end MLOps pipeline that predicts whether a customer is likely to purchase the package,
so the sales team can target outreach more effectively.

## Repository layout

```
tourism_project/
  data/
    tourism.csv                # source dataset
  model_building/
    data_register.py           # validates schema + prints a dataset summary
    prep.py                    # cleans data, splits train/test, saves CSVs
    train.py                   # tunes an XGBoost model, logs to MLflow, saves the best model
  deployment/
    app.py                     # Streamlit inference app
    requirements.txt           # dependencies for the Streamlit app
    model.joblib                # best trained model, committed by the pipeline
.github/workflows/
  pipeline.yml                  # CI/CD pipeline: register -> prep -> train -> commit model
requirements.txt                # dependencies for the ML pipeline jobs
```

## Pipeline

`.github/workflows/pipeline.yml` runs on every push to `main` (and can be triggered manually):

1. **register-dataset** — validates the dataset schema and prints a summary.
2. **data-prep** — cleans the data and creates train/test splits, passed to the next job as a
   workflow artifact.
3. **model-traning** — downloads the splits, tunes an XGBoost classifier with `GridSearchCV`,
   logs all tuned parameters and metrics to MLflow, evaluates on the test set, and commits the
   best model back to `tourism_project/deployment/model.joblib`.

## Deployment

The Streamlit app (`tourism_project/deployment/app.py`) loads the committed model and predicts
purchase likelihood from customer and sales-interaction details entered in the UI.

- **Live app:** https://tourism-package-prediction-craghav12.streamlit.app/
- **GitHub repo:** https://github.com/craghav12/tourism-package-prediction
