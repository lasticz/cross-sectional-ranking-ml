# -*- coding: utf-8 -*-
"""保存训练好的 LightGBM 模型组 + 特征列表（用于服务器部署）

产出（均保存到 user_data/ml_v2/）:
    deploy_model.pkl        LGBMRegressor  → 排序分 pred（多空排名）
    deploy_clf.pkl          LGBMClassifier → conf_dn = P(y<0)（空头门槛 >0.50）
    deploy_features.json    特征列 + 训练集中位数（在线填充 NaN）
    deploy_meta.json        训练窗口/参数/时间戳（可追溯性）

两个模型与 07_full_ls_backtest.py 的逐 fold 训练完全同参数（07: regressor 出
pred, classifier 出 conf_dn; threshold_scan.json 最优空头门槛 0.50 依赖后者）。
训练窗口: 除最后一个自然月外的全部数据（避免不完整月份的标签截断偏差）。

用法（项目根目录）:
    C:/Users/18970/.conda/envs/quant/python.exe scripts/ml_v2/08_save_deploy_model.py
"""
import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
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

# 分类器: 与 07 号 conf_dn 同参数同口径 (P(label_neutral < 0))
clf = lgb.LGBMClassifier(n_estimators=100, max_depth=6, learning_rate=0.05,
                          num_leaves=31, subsample=0.8, colsample_bytree=0.8,
                          random_state=42, verbose=-1, force_col_wise=True, n_jobs=4)
clf.fit(X_tr[valid], (y_tr[valid] > 0).astype(int))

joblib.dump(model, ML_DIR / "deploy_model.pkl")
joblib.dump(clf, ML_DIR / "deploy_clf.pkl")
# LightGBM C++ 的 save_model 在非 ASCII 路径(中文)下写文件失败 → 用纯 Python 落盘
(ML_DIR / "deploy_model.txt").write_text(model.booster_.model_to_string(), encoding="utf-8")
(ML_DIR / "deploy_features.json").write_text(json.dumps({
    "feature_cols": feat_cols,
    "fill_medians": {c: float(med[c]) for c in feat_cols},
}, indent=1), encoding="utf-8")
(ML_DIR / "deploy_meta.json").write_text(json.dumps({
    "generated_at": datetime.now(timezone.utc).isoformat(),
    "train_through": str(all_months[-2]),
    "n_features": len(feat_cols),
    "n_train_rows": int(valid.sum()),
    "regressor": {k: v for k, v in model.get_params().items()},
    "classifier": {k: v for k, v in clf.get_params().items()},
    "short_conf_threshold": 0.50,
}, indent=1, ensure_ascii=False), encoding="utf-8")

print(f"模型已保存: deploy_model.pkl / deploy_clf.pkl / deploy_model.txt")
print(f"特征: {len(feat_cols)} 个 → deploy_features.json + deploy_meta.json")
print(f"训练样本: {valid.sum()} 行 (至 {all_months[-2]})")
