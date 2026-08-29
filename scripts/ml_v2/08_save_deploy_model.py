# -*- coding: utf-8 -*-
"""保存训练好的 LightGBM 模型 + 特征列表（用于服务器部署）"""
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

ROOT = Path.cwd()
ML_DIR = ROOT / "user_data" / "ml_v2"

df = pd.read_feather(ML_DIR / "feature_matrix.feather")
meta = {"date", "base", "label_abs", "label_neutral", "label_rank"}
feat_cols = [c for c in df.columns if c not in meta]
df["date"] = pd.to_datetime(df["date"], utc=True)

months = df["date"].dt.to_period("M")
all_months = sorted(months.unique())
tr_mask = months <= all_months[-2]

med = df.loc[tr_mask, feat_cols].median(numeric_only=True)
X_tr = df.loc[tr_mask, feat_cols].fillna(med).to_numpy(dtype=np.float32)
y_tr = df.loc[tr_mask, "label_neutral"].to_numpy(dtype=np.float32)
valid = np.isfinite(y_tr)

model = lgb.LGBMRegressor(n_estimators=150, max_depth=6, learning_rate=0.05,
                           num_leaves=63, subsample=0.8, colsample_bytree=0.8,
                           random_state=42, verbose=-1, force_col_wise=True, n_jobs=4)
model.fit(X_tr[valid], y_tr[valid])

model.booster_.save_model(str(ML_DIR / "deploy_model.txt"))
(ML_DIR / "deploy_features.json").write_text(json.dumps({
    "feature_cols": feat_cols,
    "fill_medians": {c: float(med[c]) for c in feat_cols},
}, indent=1), encoding="utf-8")

print(f"模型已保存: {ML_DIR / 'deploy_model.txt'}")
print(f"特征: {len(feat_cols)} 个 → deploy_features.json")
print(f"训练样本: {valid.sum()} 行")
