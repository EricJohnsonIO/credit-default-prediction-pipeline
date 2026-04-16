from src.data_loader import get_train_test_split



from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.model_selection import (
    GridSearchCV,
    RandomizedSearchCV,
    cross_val_score,
    cross_validate,
    train_test_split,
)
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    make_scorer,
    precision_score,
    recall_score,
)
from sklearn.feature_selection import SelectFromModel
from sklearn.base import BaseEstimator, TransformerMixin

class SchemaFixer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        X = X.copy()

        X["EDUCATION"] = X["EDUCATION"].replace([0,4,5,6],0)

        X["EDUCATION"] = X["EDUCATION"].astype("category")
        X["SEX"] = X["SEX"].astype("category")
        X["MARRIAGE"] = X["MARRIAGE"].astype("category")
        return X



###

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.compose import make_column_transformer, make_column_selector


class FeatureEngineer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        bill_feats = [
            "BILL_AMT1", "BILL_AMT2", "BILL_AMT3",
            "BILL_AMT4", "BILL_AMT5", "BILL_AMT6"]
        pay_amt_feats = [
            "PAY_AMT1", "PAY_AMT2", "PAY_AMT3",
            "PAY_AMT4", "PAY_AMT5", "PAY_AMT6"]
        pay_late_feats = ["PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"]


        # Bill Feats
        X["BILL_MEAN"] = X[bill_feats].mean(axis=1)
        X["BILL_STD"] = X[bill_feats].std(axis=1)


        # PAy_amnt feats
        X["PAY_AMT_MEAN"] = X[pay_amt_feats].mean(axis=1)
        X["PAY_AMT_STD"] = X[pay_late_feats].std(axis=1)




        # pay delay
        X["DELAY_MEAN"] = X[pay_late_feats].mean(axis=1)

        X["DELAY_TREND"] = X["PAY_0"]-X["PAY_6"]
        X["DELAY_RECENT"] = X["PAY_0"]
        X["DELAY_RECENT_WORST"] = X[["PAY_0","PAY_2","PAY_3"]].max(axis=1)
        X["DELAY_MAX"] = X[pay_late_feats].max(axis=1)
        X["DELAY_RISK"] = X["DELAY_MAX"].apply(lambda x: 0 if x <= 0 else
        (1 if x <= 3 else 2)).astype("int")

        #CREDIT FEATS
        # limit bal is > 0, (0 or negative limit?
        limit = X["LIMIT_BAL"].replace(0,1)
        # upper bound, 500% over limit, is gonna be an outlier.
        X["CREDIT_UTIL"] = (X["BILL_AMT1"]/limit).clip(upper=5)

        # negative, so some sort of refund or chargeback, the kaggle wasnt super clear
        X["CREDIT_SURPLUS"] = X["BILL_AMT1"].clip(upper=0)

        # bill can sum to 0
        bill_sum = X[bill_feats].sum(axis=1).replace(0,1)
        X["REPAY_AMOUNT"] = X[pay_amt_feats].sum(axis=1)/bill_sum
        X["OVER_LIMIT"] = (X["BILL_AMT1"] > X["LIMIT_BAL"]).astype("category")
        return X


preprocessor = make_column_transformer(
    (StandardScaler(), make_column_selector(dtype_include="number")),
    (OneHotEncoder(handle_unknown="ignore"), make_column_selector(dtype_include="category")),
)



log = LogisticRegression(max_iter=10000,class_weight="balanced")
selector = SelectFromModel(LogisticRegression(max_iter=10000,class_weight="balanced",random_state=314))

X_train, X_test, y_train, y_test = get_train_test_split()
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

pipeline = Pipeline([
    ("schema", SchemaFixer()),
    ("features", FeatureEngineer()),
    ("preprocessor", preprocessor),
    ("selector", selector),
    ("model", log)
])
scores = cross_validate(pipeline, X_train, y_train, cv=cv,return_train_score=True)
print(f"log regression: "
      f"{scores['test_score'].mean():.4f} ± {scores['test_score'].std():.4f} | "
      f"train: {scores['train_score'].mean():.4f}")



dummy = LogisticRegression(max_iter=1000,class_weight="balanced")
pipeline = Pipeline([
    ("schema", SchemaFixer()),
    ("preprocessor", preprocessor),
    ("model", dummy)
])
scores = cross_validate(pipeline, X_train, y_train, cv=cv,return_train_score=True)
print(f"dummy : "
      f"{scores['test_score'].mean():.4f} ± {scores['test_score'].std():.4f} | "
      f"train: {scores['train_score'].mean():.4f}")

