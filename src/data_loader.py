
from sklearn.model_selection import (
    GridSearchCV,
    RandomizedSearchCV,
    cross_val_score,
    cross_validate,
    train_test_split,
)


import pandas as pd
from pathlib import Path
BASE_DIR = Path(__file__).resolve().parent.parent  # project root
DATA_RAW_PATH = BASE_DIR / "data" / "raw" / "UCI_Credit_Card.csv"

TARGET_COL = "default.payment.next.month"
TEST_SIZE = 0.2


def load_data():
    if not DATA_RAW_PATH.exists():
        raise FileNotFoundError("dataset not found, in data/raw, either run download.py or try to find it.")
    df = pd.read_csv(DATA_RAW_PATH, index_col=0)

    X = df.drop(columns=TARGET_COL)
    Y = df[TARGET_COL]
    return X, Y

def get_train_test_split(random_state):
    X, Y = load_data()
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=TEST_SIZE, random_state=random_state,stratify=Y)
    return X_train, X_test, y_train, y_test

