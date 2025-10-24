# save as bin/train_lgbm.py (or adapt your evaluator)
import json, joblib, pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from lightgbm import LGBMRegressor

EXCLUDE = {"trip_id","driver_id","mode","target"}
df = pd.read_csv("data/training/features.csv")
y  = df["target"].values
fe = [c for c in df.columns if c not in EXCLUDE]
X  = df[fe].values
Xtr,Xva,ytr,yva = train_test_split(X,y,test_size=0.25,random_state=42)

model = LGBMRegressor(n_estimators=600, learning_rate=0.05, subsample=0.9,
                      colsample_bytree=0.9, random_state=42)
model.fit(Xtr, ytr)
Path("models").mkdir(exist_ok=True)
joblib.dump(model, "models/lgbm_risk.joblib")
Path("models/gbm_risk_features.json").write_text(json.dumps(fe), encoding="utf-8")
print("Saved models/lgbm_risk.joblib")
