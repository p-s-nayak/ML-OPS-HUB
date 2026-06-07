import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
import joblib
import os
from huggingface_hub import HfApi, create_repo
from huggingface_hub.utils import RepositoryNotFoundError

Xtrain = pd.read_csv("hf://datasets/partha90/bank-customer-churn/Xtrain.csv")
Xtest  = pd.read_csv("hf://datasets/partha90/bank-customer-churn/Xtest.csv")
ytrain = pd.read_csv("hf://datasets/partha90/bank-customer-churn/ytrain.csv")
ytest  = pd.read_csv("hf://datasets/partha90/bank-customer-churn/ytest.csv")

numeric_features     = ['CreditScore', 'Age', 'Tenure', 'Balance',
                         'NumOfProducts', 'HasCrCard', 'IsActiveMember', 'EstimatedSalary']
categorical_features = ['Geography']

class_weight = ytrain.value_counts()[0] / ytrain.value_counts()[1]

preprocessor = make_column_transformer(
    (StandardScaler(), numeric_features),
    (OneHotEncoder(handle_unknown='ignore'), categorical_features)
)

xgb_model = xgb.XGBClassifier(scale_pos_weight=class_weight, random_state=42)

param_grid = {
    'xgbclassifier__n_estimators':     [50, 75, 100, 125, 150],
    'xgbclassifier__max_depth':         [2, 3, 4],
    'xgbclassifier__colsample_bytree':  [0.4, 0.5, 0.6],
    'xgbclassifier__colsample_bylevel': [0.4, 0.5, 0.6],
    'xgbclassifier__learning_rate':     [0.01, 0.05, 0.1],
    'xgbclassifier__reg_lambda':        [0.4, 0.5, 0.6],
}

model_pipeline = make_pipeline(preprocessor, xgb_model)
grid_search = GridSearchCV(model_pipeline, param_grid, cv=5, n_jobs=-1)
grid_search.fit(Xtrain, ytrain)

best_model = grid_search.best_estimator_
classification_threshold = 0.45

y_pred_train = (best_model.predict_proba(Xtrain)[:, 1] >= classification_threshold).astype(int)
y_pred_test  = (best_model.predict_proba(Xtest)[:, 1]  >= classification_threshold).astype(int)

print(classification_report(ytrain, y_pred_train))
print(classification_report(ytest,  y_pred_test))

joblib.dump(best_model, "best_churn_model.joblib")

repo_id   = "partha90/churn-model"
repo_type = "model"
token     = os.getenv("HF_TOKEN")
api       = HfApi(token=token)

try:
    api.repo_info(repo_id=repo_id, repo_type=repo_type)
    print(f"Model repo '{repo_id}' already exists.")
except RepositoryNotFoundError:
    create_repo(repo_id=repo_id, repo_type=repo_type, private=False, token=token)
    print(f"Model repo '{repo_id}' created.")

api.upload_file(
    path_or_fileobj="best_churn_model.joblib",
    path_in_repo="best_churn_model.joblib",
    repo_id=repo_id,
    repo_type=repo_type,
)
