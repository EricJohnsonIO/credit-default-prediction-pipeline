import joblib
import pandas as pd
from sklearn.compose import make_column_transformer, make_column_selector
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder

from src.metrics import log_experiment, print_score, print_grid_result, describe_experiment
from src.preprocessing import SchemaFixer, FeatureEngineer

from src.data_loader import get_train_test_split, load_data
import src.config as cfg
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt

from sklearn.metrics import classification_report
import shap

from sklearn.calibration import CalibrationDisplay


from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import (
    GridSearchCV,
    cross_validate,
)

from sklearn.model_selection import TunedThresholdClassifierCV
from src.experiments import *




def make_pipeline(experiment):
    preprocessor = make_column_transformer(
        (StandardScaler(), make_column_selector(dtype_include="number")),
        (OneHotEncoder(handle_unknown="ignore"), make_column_selector(dtype_include="category")),
    )
    steps = [("schema", SchemaFixer())]
    if experiment.get("FeatureEngineer", False):
        steps.append(('feature_engineer', FeatureEngineer(experiment.get("DropFeatures"))))

    steps.append(("preprocessor", preprocessor))
    if experiment.get("FeatureSelector") is not None:
        steps.append(("feature_selector", experiment.get("FeatureSelector")))
    steps.append(("model", experiment["model"]))

    return Pipeline(steps)


def make_experiment(description, model_dict, use_feature_engineering, selector=None, param_grid=None,
                    drop_features=True, calibrate=False):
    # base_rf_dict = {
    #     "name:": "RandomForestClassifier",
    #     "tuned": False,
    #     "model": RandomForestClassifier(n_estimators=100, class_weight="balanced", max_depth=5, random_state=42,
    #                                     min_samples_split=10, min_samples_leaf=5)}

    return {
        "description": description,
        "name": model_dict["name"],
        "tuned": model_dict["tuned"],
        "FeatureEngineer": use_feature_engineering,
        "FeatureSelector": selector,
        "DropFeatures": drop_features,
        "model": model_dict["model"],
        "param_grid": param_grid,
        "calibrate": calibrate}




def find_threshold(data_split):
    X_train, _, y_train, _ = data_split
    model = build_final_model()
    model = TunedThresholdClassifierCV(estimator=model, cv=5, scoring='f1')
    model.fit(X_train, y_train)
    print("Best Threshold: ", model.best_threshold_)
    print("Best Score: ", model.best_score_)
    return model.best_threshold_



def final_test(threshold, data_split):
    print(f"Results with {threshold} threshold\n", threshold)
    X_train, X_test, y_train, y_test = data_split
    model = build_final_model()
    model.fit(X_train, y_train)


    train_scores = model.predict_proba(X_train)[:, 1]
    test_scores = model.predict_proba(X_test)[:, 1]
    y_pred = (test_scores >= threshold).astype("int")

    print("Train roc_auc-AUC:", roc_auc_score(y_train, train_scores))
    print("Test roc-auc:", roc_auc_score(y_test, test_scores))

    ConfusionMatrixDisplay.from_predictions(y_test, y_pred,
                                                   display_labels=["wont default next month",
                                                                   "will default next month"],
                                                   values_format="d",
                                                   cmap="Blues",
                                                   colorbar=False,
                                                   )
    print(y_pred.shape)
    plt.savefig(cfg.CONFUSION_MATRIX_PATH, bbox_inches='tight', dpi=300)

    plt.show()
    cr_dict = classification_report(
        y_test, y_pred, target_names=["wont default payment", "will default payment"],
        output_dict=True)

    cr = pd.DataFrame(cr_dict).T

    print(cr.loc[["wont default payment", "will default payment"]])

    explainer = shap.Explainer(model.predict_proba, X_train.iloc[:100])
    shap_values = explainer(X_test.iloc[:100])

    shap.plots.beeswarm(shap_values[:, :, 1], show=False)
    plt.savefig(cfg.BEESWARM_PATH, bbox_inches='tight', dpi=300)
    plt.close()
    shap.plots.waterfall(shap_values[0, :, 1], show=False)
    plt.savefig(cfg.WATERFALL_PATH, bbox_inches='tight', dpi=300)
    plt.close()

    fig,ax = plt.subplots()
    CalibrationDisplay.from_estimator(model,X_test,y_test,n_bins=10,name="Calibrated",ax=ax)

    pipeline = model.estimator
    pipeline.fit(X_train, y_train)

    CalibrationDisplay.from_estimator(pipeline,X_test,y_test,n_bins=10,name="Uncalibrated",ax=ax)
    ax.legend()
    plt.savefig(cfg.CALIBRATION_CURVE_PATH, bbox_inches='tight', dpi=300)


def build_final_model() -> CalibratedClassifierCV:
    exp = make_experiment("", hgb_tuned_dict, True, drop_features=False)
    pipeline = make_pipeline(exp)
    model = CalibratedClassifierCV(estimator=pipeline, method="sigmoid", cv=5)
    return model

def save_model():
    model = build_final_model()
    X, y = load_data()
    model.fit(X,y)
    joblib.dump(model, cfg.MODEL_PATH)

def main():
    X_train, X_test, y_train, y_test = get_train_test_split(cfg.SEED)
    data_split = (X_train, X_test, y_train, y_test)
    experiments = [
        # make_experiment("", base_rf_dict, False),
        # make_experiment("", base_rf_dict, True, tree_selector),
        # make_experiment("", lr_dict, False),
        # make_experiment("", lr_dict, True, log_selector),
        # make_experiment("", hist_grad_dict, True),
        # make_experiment("", base_rf_dict, True,tree_selector,drop_features=False),

        # # FE is helping, need to test hist with FS
        # # RF and Hist are best so far, so I'll ignore lr
        # # RF benefited from both raw and engineered features
        # # Hgd is slightly overfitting, will do param search
        #
        # make_experiment("", hist_grad_dict, True,drop_features=False),
        # # slightly improved, so will run param search with hist
        #
        # # make_experiment("", hist_grad_dict, False, param_grid=hist_grid, drop_features=False),
        #
        # # make_experiment("", hist_grad_dict, False, param_grid=hist_grid_v2, drop_features=False),
        # # all the top models are in std of each other for the two scores.
        # # so ill pick the least likely to overfit, while leaning towards Roc-Auc over avg precision
        #
        # make_experiment("", hgb_tuned_dict, True, drop_features=False),
        # # within std of default, but this one is a lot more regularized.

        # make_experiment("", lr_dict, True, param_grid=log_grid, drop_features=True),
        # under performs compared to hist

        # make_experiment("", calib_hgb_dict, True, drop_features=False,calibrate=True),

    ]
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=cfg.SEED)
    scoring = ["roc_auc", "average_precision", "f1", "recall", "precision", "accuracy"]
    # adjustments to my original grid search, n est 300 was maxed out, also going to focus around min sample =10
    # balanced was better in all the top cases
    # I originally forgot min sample split

    for exp in experiments:
        pipeline = make_pipeline(exp)
        describe_experiment(exp)
        if exp.get("param_grid") is not None:
            grid = GridSearchCV(pipeline, exp.get("param_grid"), cv=cv, scoring=scoring, n_jobs=-1, refit="roc_auc")
            grid.fit(X_train, y_train)
            print_grid_result(grid, exp.get("param_grid"))
        elif exp.get("calibrate"):
            model = CalibratedClassifierCV(pipeline, method="sigmoid", cv=cv)
            scores = cross_validate(model, X_train, y_train, cv=cv, return_train_score=True, scoring=scoring)
            print_score(scores)
        else:
            scores = cross_validate(pipeline, X_train, y_train, cv=cv, return_train_score=True, scoring=scoring)
            print_score(scores)
            log_experiment(exp, scores)

    best_f1_threshold = find_threshold(data_split)
    # best_f1_threshold = 0.2877729048920662
    # Best Threshold: 0.2877729048920662
    # Best Score: 0.5465156161489235


    # final_test(0.5, data_split)

    final_test(best_f1_threshold, data_split)


    save_model()




if __name__ == '__main__':
    main()
