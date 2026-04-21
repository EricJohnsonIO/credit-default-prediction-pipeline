from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LogisticRegression
import src.config as cfg


lr_dict = {
    "name": "Logistic Regression",
    "tuned": False,
    "model": LogisticRegression(max_iter=10000, class_weight="balanced")}

tree_selector = SelectFromModel(RandomForestClassifier(n_estimators=100, max_depth=10, random_state=cfg.SEED),
                                threshold="median")
log_selector = SelectFromModel(LogisticRegression(max_iter=10000, class_weight="balanced", random_state=cfg.SEED),
                               threshold="median")
log_grid = {
    "model__C": [0.001, 0.01, 0.1, 1, 10],
    "model__random_state": [cfg.SEED],
    "model__solver": ["lbfgs"],
    "model__l1_ratio": [0],
    "model__class_weight": [None, "balanced"],
}
base_rf_dict = {
    "name": "RandomForest",
    "tuned": False,
    "model": RandomForestClassifier(n_estimators=100, class_weight="balanced", max_depth=5, random_state=cfg.SEED,
                                    min_samples_split=10, min_samples_leaf=5)}
tuned_rf_dict = {
    "name": "RandomForest",
    "tuned": True,
    "model": RandomForestClassifier(n_estimators=300, class_weight="balanced", max_depth=20, random_state=cfg.SEED,
                                    min_samples_split=10, min_samples_leaf=5)}
rf_grid = {
    "model__n_estimators": [100, 200],
    "model__max_depth": [None, 5, 10],
    "model__min_samples_leaf": [1, 5, 10],
    "model__max_features": ["sqrt", "log2"],
    "model__class_weight": ["balanced"],
    "model__min_samples_split": [2, 10],
}
rf_grid_v2 = {
    "model__n_estimators": [100, 300, 500],
    "model__max_depth": [None, 20],
    "model__min_samples_leaf": [5, 10, 20],
    "model__max_features": ["sqrt", "log2"],
    "model__class_weight": ["balanced"],
    "model__min_samples_split": [2, 10, 20],
}

hist_grad_dict = {
    "name": "HistGradientBoosting",
    "tuned": False,
    "model": HistGradientBoostingClassifier(class_weight="balanced", max_depth=5, random_state=cfg.SEED, min_samples_leaf=5)
}

hgb_tuned_dict = {
    "name": "HistGradientBoosting",
    "tuned": True,
    "model": HistGradientBoostingClassifier(class_weight="balanced", learning_rate=0.05, max_depth=None, max_iter=200,
                                            random_state=cfg.SEED, min_samples_leaf=500, l2_regularization=0)
}
hist_grid_v2 = {
    "model__learning_rate": [0.05, 0.1],
    "model__max_depth": [None, 3, 5],
    "model__max_iter": [200],
    "model__min_samples_leaf": [20, 100, 150, 200, 300, 500],
    "model__l2_regularization": [0, 1],
}
hist_grid = {
    "model__learning_rate": [0.01, 0.05, 0.1],
    "model__max_depth": [None, 3, 5, 10],
    "model__max_iter": [100, 200, 300],
    "model__min_samples_leaf": [5, 10, 20, 100],
    "model__l2_regularization": [0, 0.1, 1],
}

calib_hgb_dict = {
    "name": "Calibrated HistGradientBoosting",
    "tuned": True,
    "model": HistGradientBoostingClassifier(class_weight="balanced", learning_rate=0.05, max_depth=None, max_iter=200,
                                            random_state=cfg.SEED, min_samples_leaf=500, l2_regularization=0)
}
tuned_hgbc = HistGradientBoostingClassifier(class_weight="balanced", learning_rate=0.05, max_depth=None, max_iter=200,
                                            min_samples_leaf=500, l2_regularization=0)
