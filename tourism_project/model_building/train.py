"""
Loads the train/test splits produced by prep.py, tunes an XGBoost
classifier with GridSearchCV, tracks the experiment with MLflow, evaluates
the best model, and saves it so the pipeline can commit it to the repo.

Run from the repository root, after prep.py has produced Xtrain.csv,
Xtest.csv, ytrain.csv and ytest.csv:
    python tourism_project/model_building/train.py
"""

import os

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
import xgboost as xgb
from sklearn.compose import make_column_transformer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

MODEL_OUT_PATH = os.path.join("tourism_project", "deployment", "model.joblib")

CATEGORICAL_FEATURES = [
    "TypeofContact",
    "Occupation",
    "Gender",
    "ProductPitched",
    "MaritalStatus",
    "Designation",
]

PARAM_GRID = {
    "xgbclassifier__n_estimators": [100, 200],
    "xgbclassifier__max_depth": [3, 5],
    "xgbclassifier__learning_rate": [0.05, 0.1],
}


def load_splits():
    Xtrain = pd.read_csv("Xtrain.csv")
    Xtest = pd.read_csv("Xtest.csv")
    ytrain = pd.read_csv("ytrain.csv").squeeze("columns")
    ytest = pd.read_csv("ytest.csv").squeeze("columns")
    return Xtrain, Xtest, ytrain, ytest


def build_pipeline():
    preprocessor = make_column_transformer(
        (OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
        remainder=StandardScaler(),
    )
    model = xgb.XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=42,
    )
    return make_pipeline(preprocessor, model)


def train():
    Xtrain, Xtest, ytrain, ytest = load_splits()
    pipeline = build_pipeline()

    mlflow.set_experiment("tourism-wellness-package-prediction")

    with mlflow.start_run(run_name="xgboost_gridsearch") as parent_run:
        grid_search = GridSearchCV(
            pipeline,
            param_grid=PARAM_GRID,
            scoring="roc_auc",
            cv=3,
            n_jobs=-1,
        )
        grid_search.fit(Xtrain, ytrain)

        # Log every tuned parameter combination as its own nested run.
        cv_results = grid_search.cv_results_
        for i, params in enumerate(cv_results["params"]):
            with mlflow.start_run(run_name=f"candidate_{i}", nested=True):
                mlflow.log_params(params)
                mlflow.log_metric("mean_cv_roc_auc", cv_results["mean_test_score"][i])
                mlflow.log_metric("std_cv_roc_auc", cv_results["std_test_score"][i])

        best_pipeline = grid_search.best_estimator_
        mlflow.log_params(grid_search.best_params_)
        mlflow.log_metric("best_cv_roc_auc", grid_search.best_score_)

        # Evaluate on the held-out test set.
        y_pred = best_pipeline.predict(Xtest)
        y_proba = best_pipeline.predict_proba(Xtest)[:, 1]

        test_metrics = {
            "test_accuracy": accuracy_score(ytest, y_pred),
            "test_precision": precision_score(ytest, y_pred),
            "test_recall": recall_score(ytest, y_pred),
            "test_f1": f1_score(ytest, y_pred),
            "test_roc_auc": roc_auc_score(ytest, y_proba),
        }
        mlflow.log_metrics(test_metrics)
        mlflow.sklearn.log_model(best_pipeline, artifact_path="model")

        print("Best hyperparameters:", grid_search.best_params_)
        print("Best CV ROC-AUC     :", grid_search.best_score_)
        print("-" * 60)
        print("Test set classification report:")
        print(classification_report(ytest, y_pred))
        print("Test set metrics:", test_metrics)

        os.makedirs(os.path.dirname(MODEL_OUT_PATH), exist_ok=True)
        joblib.dump(best_pipeline, MODEL_OUT_PATH)
        print(f"Best model saved to {MODEL_OUT_PATH}")

    return best_pipeline, test_metrics


if __name__ == "__main__":
    train()
