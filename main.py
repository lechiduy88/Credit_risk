import joblib  
from pathlib import Path  
import gc  
from glob import glob  
import numpy as np  
import pandas as pd  
import polars as pl  
from sklearn.base import BaseEstimator, RegressorMixin
from imblearn.over_sampling import RandomOverSampler 
from imblearn.under_sampling import RandomUnderSampler   
from sklearn.metrics import roc_auc_score  
import lightgbm as lgb  
import warnings  
from sklearn.model_selection import train_test_split
from sklearn.model_selection import TimeSeriesSplit, GroupKFold, StratifiedGroupKFold

warnings.filterwarnings('ignore')  

ROOT = '/kaggle/input/home-credit-credit-risk-model-stability'

class Pipeline:
    def set_table_dtypes(df):
        for col in df.columns:
            if col in ["case_id", "WEEK_NUM", "num_group1", "num_group2"]:
                df = df.with_columns(pl.col(col).cast(pl.Int64))
            elif col in ["date_decision"]:
                df = df.with_columns(pl.col(col).cast(pl.Date))
            elif col[-1] in ("P", "A"):
                df = df.with_columns(pl.col(col).cast(pl.Float64))
            elif col[-1] in ("M",):
                df = df.with_columns(pl.col(col).cast(pl.String))
            elif col[-1] in ("D",):
                df = df.with_columns(pl.col(col).cast(pl.Date))
        return df

    def handle_dates(df):
        for col in df.columns:
            if col[-1] in ("D",):
                df = df.with_columns(pl.col(col) - pl.col("date_decision"))  
                df = df.with_columns(pl.col(col).dt.total_days())  
        df = df.drop("date_decision", "MONTH")  
        return df
    
    def filter_cols(df):
        for col in df.columns:
            if col not in ["target", "case_id", "WEEK_NUM"]:
                isnull = df[col].is_null().mean()
                if isnull > 0.7:
                    df = df.drop(col)  
        
        for col in df.columns:
            if (col not in ["target", "case_id", "WEEK_NUM"]) & (df[col].dtype == pl.String):
                freq = df[col].n_unique()
                if (freq == 1) | (freq > 200):
                    df = df.drop(col)  
        
        return df

class Aggregator:
    def num_expr(df):
        cols = [col for col in df.columns if col[-1] in ("P", "A")]  
        expr_max = [pl.max(col).alias(f"max_{col}") for col in cols]  
        expr_last = [pl.last(col).alias(f"last_{col}") for col in cols] 
        expr_mean = [pl.mean(col).alias(f"mean_{col}") for col in cols]  
        expr_median = [pl.median(col).alias(f"median_{col}") for col in cols] 
        return expr_max + expr_last + expr_mean 

    def date_expr(df):
        cols = [col for col in df.columns if col[-1] in ("D")]  
        expr_max = [pl.max(col).alias(f"max_{col}") for col in cols]  
        expr_last = [pl.last(col).alias(f"last_{col}") for col in cols]  
        expr_mean = [pl.mean(col).alias(f"mean_{col}") for col in cols]  
        expr_median = [pl.median(col).alias(f"median_{col}") for col in cols] 
        return expr_max + expr_last + expr_mean 

    def str_expr(df):
        cols = [col for col in df.columns if col[-1] in ("M",)]  
        expr_max = [pl.max(col).alias(f"max_{col}") for col in cols]  
        expr_last = [pl.last(col).alias(f"last_{col}") for col in cols]  
        return expr_max + expr_last

    def other_expr(df):
        cols = [col for col in df.columns if col[-1] in ("T", "L")]  
        expr_max = [pl.max(col).alias(f"max_{col}") for col in cols]  
        expr_last = [pl.last(col).alias(f"last_{col}") for col in cols]  
        return expr_max + expr_last

    def count_expr(df):
        cols = [col for col in df.columns if "num_group" in col] 
        expr_max = [pl.max(col).alias(f"max_{col}") for col in cols]  
        expr_last = [pl.last(col).alias(f"last_{col}") for col in cols]  
        return expr_max + expr_last

    def get_exprs(df):
        exprs = Aggregator.num_expr(df) + \
                Aggregator.date_expr(df) + \
                Aggregator.str_expr(df) + \
                Aggregator.other_expr(df) + \
                Aggregator.count_expr(df)
        return exprs
    

def read_file(path, depth=None):
    df = pl.read_parquet(path)
    df = df.pipe(Pipeline.set_table_dtypes)
    if depth in [1, 2]:
        df = df.group_by("case_id").agg(Aggregator.get_exprs(df)) 
    return df

def read_files(regex_path, depth=None):
    chunks = []
    for path in glob(str(regex_path)):
        df = pl.read_parquet(path)
        df = df.pipe(Pipeline.set_table_dtypes)
        if depth in [1, 2]:
            df = df.group_by("case_id").agg(Aggregator.get_exprs(df))
        chunks.append(df)
    df = pl.concat(chunks, how="vertical_relaxed").unique(subset=["case_id"])
    return df

def feature_eng(df_base, depth_0, depth_1, depth_2):
    df_base = df_base.with_columns(
        month_decision = pl.col("date_decision").dt.month(),
        weekday_decision = pl.col("date_decision").dt.weekday(),
    )
    for i, df in enumerate(depth_0 + depth_1 + depth_2):
        df_base = df_base.join(df, how="left", on="case_id", suffix=f"_{i}")
    df_base = df_base.pipe(Pipeline.handle_dates)
    return df_base

def to_pandas(df_data, cat_cols=None):
    df_data = df_data.to_pandas()
    if cat_cols is None:
        cat_cols = list(df_data.select_dtypes("object").columns)
    df_data[cat_cols] = df_data[cat_cols].astype("category")
    return df_data, cat_cols

def reduce_mem_usage(df):
    """ 
    Lặp qua tất cả các cột của dataframe và sửa đổi kiểu dữ liệu
    để giảm mức sử dụng bộ nhớ.
    """
    start_mem = df.memory_usage().sum() / 1024**2  # Mức sử dụng bộ nhớ trước khi tối ưu hóa
    print('Mức sử dụng bộ nhớ của dataframe là {:.2f} MB'.format(start_mem))
    
    for col in df.columns:
        col_type = df[col].dtype
        if str(col_type)=="category":
            continue

        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
        else:
            continue
    end_mem = df.memory_usage().sum() / 1024**2  # Mức sử dụng bộ nhớ sau khi tối ưu hóa
    print('Mức sử dụng bộ nhớ sau khi tối ưu hóa là: {:.2f} MB'.format(end_mem))
    print('Giảm {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))
    
    return df



ROOT = Path("D:\DATN\home-credit-credit-risk-model-stability")
TRAIN_DIR = ROOT / "parquet_files" / "train"

data_store = {
    "df_base": read_file(TRAIN_DIR / "train_base.parquet"),
    "depth_0": [
        read_file(TRAIN_DIR / "train_static_cb_0.parquet"),
        read_files(TRAIN_DIR / "train_static_0_*.parquet"),
    ],
    "depth_1": [
        read_files(TRAIN_DIR / "train_applprev_1_*.parquet", 1),
        read_file(TRAIN_DIR / "train_tax_registry_a_1.parquet", 1),
        read_file(TRAIN_DIR / "train_tax_registry_b_1.parquet", 1),
        read_file(TRAIN_DIR / "train_tax_registry_c_1.parquet", 1),
        read_files(TRAIN_DIR / "train_credit_bureau_a_1_*.parquet", 1),
        read_file(TRAIN_DIR / "train_credit_bureau_b_1.parquet", 1),
        read_file(TRAIN_DIR / "train_other_1.parquet", 1),
        read_file(TRAIN_DIR / "train_person_1.parquet", 1),
        read_file(TRAIN_DIR / "train_deposit_1.parquet", 1),
        read_file(TRAIN_DIR / "train_debitcard_1.parquet", 1),
    ],
    "depth_2": [
        read_file(TRAIN_DIR / "train_credit_bureau_b_2.parquet", 2),
        read_files(TRAIN_DIR / "train_credit_bureau_a_2_*.parquet", 2),
        read_file(TRAIN_DIR / "train_applprev_2.parquet", 2),
        read_file(TRAIN_DIR / "train_person_2.parquet", 2)
    ]
}

df = feature_eng(**data_store)
print("train data shape:\t", df.shape)

df = df.pipe(Pipeline.filter_cols)
print("train data shape:\t", df.shape)


df, cat_cols = to_pandas(df)
df = reduce_mem_usage(df)

del data_store
gc.collect()

params = {
    "boosting_type": "gbdt",
    "objective": "binary",
    "metric": "auc",
    "max_depth": 10,
    "learning_rate": 0.05,
    "max_bin": 255,
    "n_estimators": 1000,
    "colsample_bytree": 0.8,
    "colsample_bynode": 0.8,
    "random_state": 42,
    "reg_alpha": 0.1,
    "reg_lambda": 10,
    "extra_trees": True,
    'num_leaves': 64,
    "device": "cpu",
    "verbose" : -1 
}

fitted_models = []
def gini_stability(base, w_fallingrate=88.0, w_resstd=-0.5):
    gini_in_time = base.loc[:, ["WEEKS_NUM", "target", "score"]]\
        .sort_values("WEEKS_NUM")\
        .groupby("WEEKS_NUM")[["target", "score"]]\
        .apply(lambda x: 2*roc_auc_score(x["target"], x["score"])-1).tolist()
    x = np.arange(len(gini_in_time))
    y = np.array(gini_in_time)
    a, b = np.polyfit(x, y, 1)
    y_hat = a*x + b
    residuals = y - y_hat
    res_std = np.std(residuals)
    avg_gini = np.mean(gini_in_time)
    return avg_gini + w_fallingrate * min(0, a) + w_resstd * res_std


from sklearn.utils import resample
from sklearn.utils.class_weight import compute_sample_weight


weeks = df["WEEK_NUM"].unique()
sampled_data = pd.DataFrame()
def sample_week_data(df, target_value, week, sample_rate):
    week_data = df[(df["target"] == target_value) & (df["WEEK_NUM"] == week)]
    sampled_week = resample(week_data, n_samples=int(len(week_data) * sample_rate))
    return sampled_week

for week in weeks:
    sampled_max_week = sample_week_data(df, 0, week, 3/4)
    sampled_data = pd.concat([sampled_data, sampled_max_week])

for week in weeks:
    sampled_minority_week = sample_week_data(df, 1, week, 10)
    sampled_data = pd.concat([sampled_data, sampled_minority_week])

sample_weights = compute_sample_weight(class_weight='balanced', y=sampled_data["target"])
sample_weights = pd.Series(sample_weights, index=sampled_data.index)

sampled_data = sampled_data.sample(n=len(sampled_data), weights=sample_weights, random_state=42)


test_weeks = weeks[np.argsort(-weeks)][:int(len(weeks) * 0.1)]
train = sampled_data[~sampled_data["WEEK_NUM"].isin(test_weeks)]
test = sampled_data[sampled_data["WEEK_NUM"].isin(test_weeks)]

print(train.shape)
X = train.drop(columns=["target", "case_id", "WEEK_NUM"])
y = train["target"]
weeks = train["WEEK_NUM"]


cv = StratifiedGroupKFold(n_splits=5, shuffle=False)
gini_stability_scores_valid = []
import matplotlib.pyplot as plt

for idx_train, idx_valid in cv.split(X, y, groups=weeks):
    X_train, y_train = X.iloc[idx_train], y.iloc[idx_train]
    X_valid, y_valid = X.iloc[idx_valid], y.iloc[idx_valid]
    print(f"Shape of training data with label 1: {y_train[y_train == 1].shape[0]}, Shape of training data with label 0: {y_train[y_train == 0].shape[0]}")
    print(f"Shape of validation data with label 1: {y_valid[y_valid == 1].shape[0]}, Shape of validation data with label 0: {y_valid[y_valid == 0].shape[0]}")
    model = lgb.LGBMClassifier(**params)
    model.fit(
        X_train, y_train,
        eval_set=[(X_valid, y_valid)],
        callbacks=[lgb.log_evaluation(50), lgb.early_stopping(50)]
    )
    y_pred_valid_proba = model.predict_proba(X_valid)[:, 1]
    auc_score_valid = roc_auc_score(y_valid, y_pred_valid_proba)

    # Tính toán gini stability cho tập validation
    base_valid = X_valid.copy()
    base_valid["target"] = y_valid
    base_valid["score"] = y_pred_valid_proba
    base_valid["WEEKS_NUM"] = weeks.iloc[idx_valid]
    stability_score = gini_stability(base_valid)
    print(f"Gini stability score for this fold: {stability_score}")
    gini_stability_scores_valid.append(stability_score)



for i, score in enumerate(gini_stability_scores_valid, start=1):
    print(f"Gini stability score for validation set {i}: {score}")

test_base = pd.DataFrame({"WEEKS_NUM": test["WEEK_NUM"], "target": test["target"], "score": model.predict_proba(test.drop(columns=["target", "case_id", "WEEK_NUM"]))[:, 1]})
test_stability_score = gini_stability(test_base)
print(f"Gini stability score for test set: {test_stability_score}")