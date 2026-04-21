
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

class FeatureEngineer(BaseEstimator, TransformerMixin):
    def __init__(self,drop_raws=True):
        self.drop_raws = drop_raws

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
        X["PAY_AMT_STD"] = X[pay_amt_feats].std(axis=1)




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

        if self.drop_raws:
            X = X.drop(columns=bill_feats + pay_amt_feats +pay_late_feats)
        return X



