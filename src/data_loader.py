import os

from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from sklearn.model_selection import (
    GridSearchCV,
    RandomizedSearchCV,
    cross_val_score,
    cross_validate,
    train_test_split,
)

from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

import numpy as np
import pandas as pd
from pathlib import Path
BASE_DIR = Path(__file__).resolve().parent.parent  # project root
DATA_RAW_PATH = BASE_DIR / "data" / "raw" / "UCI_Credit_Card.csv"

TARGET_COL = "default.payment.next.month"
SEED = 314
TEST_SIZE = 0.2


def test_path():
    print(os.getcwd())
    print(BASE_DIR)
    print("Exists?", DATA_RAW_PATH.exists())

def load_data():
    df = pd.read_csv(DATA_RAW_PATH, index_col=0)

    X = df.drop(columns=TARGET_COL)
    Y = df[TARGET_COL]
    return X, Y

def get_train_test_split():
    X, Y = load_data()
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=TEST_SIZE, random_state=SEED,stratify=Y)
    return X_train, X_test, y_train, y_test

