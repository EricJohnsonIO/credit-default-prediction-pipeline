import pandas as pd

from src import config as cfg


def log_experiment(exp, scores):
    # "description": description,
    # "name": model_dict["name"],
    # "tuned": model_dict["tuned"],
    # "FeatureEngineer": use_feature_engineering,
    # "FeatureSelector": selector,
    # "DropFeatures": drop_features,
    # "model": model_dict["model"],
    # "param_grid": param_grid,
    # "calibrate": calibrate}
    row = {
        "Model": exp.get("name"),
        "FeatureEngineering": exp.get("FeatureEngineer"),
        "DropFeatures": exp.get("DropFeatures"),
        "FeatureSelector": exp.get("FeatureSelector"),
        "tuned": exp.get("tuned"),
        "Roc_Auc": scores['test_roc_auc'].mean(),
        "Avg Precision": scores['test_average_precision'].mean(),
        "F1": scores['test_f1'].mean(),
        "Recall": scores['test_recall'].mean(),
        "Precision": scores['test_precision'].mean(),
    }
    df = pd.DataFrame([row])
    df.to_csv(cfg.LOG_PATH, mode='a', index=False, header=not cfg.LOG_PATH.exists())


def print_score(scores):
    print(
        f"Roc_Auc:      {scores['test_roc_auc'].mean():.4f} ± {scores['test_roc_auc'].std():.4f} | "
        f"train:        {scores['train_roc_auc'].mean():.4f}\n"
        f"Avg Precision:{scores['test_average_precision'].mean():.4f} ± {scores['test_average_precision'].std():.4f} | "
        f"F1:           {scores['test_f1'].mean():.4f} ± {scores['test_f1'].std():.4f} | "
        f"Recall:       {scores['test_recall'].mean():.4f} ± {scores['test_recall'].std():.4f}\n"
        f"Precision:    {scores['test_precision'].mean():.4f} ± {scores['test_precision'].std():.4f}"
    )
    # dict = {"roc_auc_mean": scores['test_roc_auc'].mean(),
    #         "roc_auc_std": scores['test_roc_auc'].std(),
    #         "train": scores['train_roc_auc'].mean(),
    #         "Avg Precision": scores['test_average_precision'].mean(),
    #         "Avg Precision std": scores['test_average_precision'].std(),
    #         "F1": scores['test_f1'].mean(),
    #         "Recall": scores['test_recall'].mean(),
    #         "Precision": scores['test_precision'].mean()}
    # df = pd.DataFrame([dict])
    # PATH = "scores.csv"
    # if not os.path.exists(PATH):
    #     df.to_csv(PATH, index=False)
    # else:
    #     df.to_csv(PATH, mode="a", index=False)


def print_grid_result(grid, param_grid, top_n=10):
    results = pd.DataFrame(grid.cv_results_)
    results["rank_roc"] = results["mean_test_roc_auc"].rank(ascending=False)
    results["rank_ap"] = results["mean_test_average_precision"].rank(ascending=False)

    base_cols = [
        "mean_test_roc_auc",
        "std_test_roc_auc",
        "mean_test_average_precision",
        "mean_test_f1",
        "mean_test_recall",
        "mean_test_precision",
        "rank_roc",
        "rank_ap",
    ]
    param_cols = [f"param_{k}" for k in param_grid.keys()]
    cols = base_cols + param_cols

    print(
        results[cols]
        .sort_values("mean_test_roc_auc", ascending=False)
        .head(top_n)
        .to_string(index=False)
    )
    print(
        results[cols]
        .sort_values("mean_test_average_precision", ascending=False)
        .head(top_n)
        .to_string(index=False)
    )
    print("\n Best Roc_Auc: ", grid.best_score_)
    print("\n Best Params: ", grid.best_params_)


def describe_experiment(experiment):
    FEATURE_SELECTOR_TAG = "+ FS"
    FEATURE_ENGINEER_TAG = "- FE"
    GRID_TAG = "GRID SEARCH"
    # "description": description,
    # "name": model_dict["name"],
    # "tuned": model_dict["tuned"],
    # "FeatureEngineer": use_feature_engineering,
    # "FeatureSelector": selector,
    # "model": model_dict["model"],
    # "param_grid": param_grid}
    parts = []
    parts.append(experiment.get("description", ""))

    if experiment.get("calibrate"):
        parts.append("Calibrated")

    parts.append(experiment.get("name", "NAME NOT FOUND"))

    if experiment.get("param_grid") is not None:
        parts.append(GRID_TAG)
    elif experiment.get("tuned") is False:
        # parts.append("Tuned")
        parts.append("(Baseline)")

    if experiment.get("FeatureEngineer"):
        parts.append(FEATURE_ENGINEER_TAG)
        if experiment.get("DropFeatures"):
            parts.append("(Dropped Source Features)")
        else:
            parts.append("(Keep Source Features)")

    if experiment.get("FeatureSelector") is not None:
        parts.append(FEATURE_SELECTOR_TAG)

    full_description = " ".join(parts)
    print(f"\n{full_description}\n")
