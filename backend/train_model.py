import os
import json
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "..", "data", "diabetic_data.csv")

df = pd.read_csv(DATA_PATH)
df = df.replace("?", np.nan)
df["readmitted_binary"] = (df["readmitted"] == "<30").astype(int)

FEATURES = [
    "age", "gender", "insulin",
    "time_in_hospital", "num_medications",
    "number_inpatient", "number_emergency",
]
CATEGORICAL = ["age", "gender", "insulin"]
NUMERIC = ["time_in_hospital", "num_medications", "number_inpatient", "number_emergency"]

X = df[FEATURES].dropna()
y = df.loc[X.index, "readmitted_binary"]

preprocessor = ColumnTransformer([
    ("cat", OneHotEncoder(drop="first", handle_unknown="ignore", sparse_output=False), CATEGORICAL),
    ("num", StandardScaler(), NUMERIC),
])

pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42)),
])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)

print("=== Model Evaluation ===")
print(classification_report(y_test, y_pred))

ohe = pipeline.named_steps["preprocessor"].named_transformers_["cat"]
cat_names = ohe.get_feature_names_out(CATEGORICAL).tolist()
all_names = cat_names + NUMERIC
coefs = pipeline.named_steps["classifier"].coef_[0].tolist()

importance = [
    {"feature": name, "coefficient": round(coef, 4)}
    for name, coef in zip(all_names, coefs)
]
importance.sort(key=lambda x: abs(x["coefficient"]), reverse=True)

importance_path = os.path.join(BASE_DIR, "feature_importance.json")
with open(importance_path, "w") as f:
    json.dump(importance, f, indent=2)

model_path = os.path.join(BASE_DIR, "model_pipeline.pkl")
joblib.dump(pipeline, model_path)

print(f"\nSaved model  -> {model_path}")
print(f"Saved importance -> {importance_path}")
print("\nTop 10 features by coefficient magnitude:")
for item in importance[:10]:
    print(f"  {item['feature']:40s}  {item['coefficient']:+.4f}")
