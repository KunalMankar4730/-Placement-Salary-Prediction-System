import pandas as pd
import numpy as np
import sqlite3
import pickle
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing   import StandardScaler
from sklearn.linear_model    import LogisticRegression
from sklearn.tree            import DecisionTreeClassifier
from sklearn.ensemble        import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics         import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    mean_squared_error, mean_absolute_error, r2_score
)

print("Starting Training...")

# ============================================================
# STEP 1 - Load the CSV file
# ============================================================
df = pd.read_csv("student_placement.csv")
print("Data loaded. Total rows:", len(df))

# ============================================================
# STEP 2 - Save data to SQLite database
# ============================================================
conn = sqlite3.connect("placement.db")
df.to_sql("student_data", conn, if_exists="replace", index=False)
conn.close()
print("Data saved to placement.db")

# ============================================================
# STEP 3 - Clean and prepare data
# ============================================================

# Fill missing values
df["extracurricular_involvement"] = df["extracurricular_involvement"].fillna("Unknown")

# Convert placement_status to 0 and 1
df["placement_status"] = df["placement_status"].map({
    "Placed"    : 1,
    "Not Placed": 0
})

# Check if mapping worked
if df["placement_status"].isnull().sum() > 0:
    print("ERROR: placement_status has null values after mapping!")
else:
    print("placement_status mapped successfully")

print("Placed students  :", df["placement_status"].sum())
print("Not placed       :", (df["placement_status"] == 0).sum())

# ============================================================
# STEP 4 - Select features
# ============================================================
features = [
    "cgpa",
    "internships_completed",
    "projects_completed",
    "coding_skill_rating",
    "communication_skill_rating",
    "aptitude_skill_rating"
]

X = df[features]
y = df["placement_status"]

# ============================================================
# STEP 5 - Split data first, then scale
# Note: We split BEFORE scaling to avoid data leakage.
# Data leakage means test data influencing the training process.
# ============================================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

scaler     = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)  # fit only on train data
X_test_sc  = scaler.transform(X_test)       # only transform test data

# Save scaler
pickle.dump(scaler, open("scaler.pkl", "wb"))
print("Scaler saved")

# ============================================================
# STEP 6 - Train placement models and compare them
# ============================================================
models = {
    "Logistic Regression": LogisticRegression(max_iter=2000, random_state=42),
    "Decision Tree"      : DecisionTreeClassifier(max_depth=6, random_state=42),
    "Random Forest"      : RandomForestClassifier(
                               n_estimators=100,
                               max_depth=10,
                               class_weight="balanced",
                               random_state=42
                           ),
}

results    = []
best_acc   = 0
best_model = None
best_name  = ""

for name, model in models.items():
    model.fit(X_train_sc, y_train)

    preds = model.predict(X_test_sc)
    proba = model.predict_proba(X_test_sc)[:, 1]

    acc = accuracy_score(y_test, preds)
    pre = precision_score(y_test, preds, zero_division=0)
    rec = recall_score(y_test, preds, zero_division=0)
    f1  = f1_score(y_test, preds, zero_division=0)
    auc = roc_auc_score(y_test, proba)

    print(f"{name} -> Accuracy: {round(acc, 3)}  AUC: {round(auc, 3)}")

    results.append({
        "Model"    : name,
        "Accuracy" : round(acc, 4),
        "Precision": round(pre, 4),
        "Recall"   : round(rec, 4),
        "F1"       : round(f1,  4),
        "ROC-AUC"  : round(auc, 4),
    })

    if acc > best_acc:
        best_acc   = acc
        best_model = model
        best_name  = name

# ============================================================
# STEP 7 - Save best placement model
# ============================================================
if not os.path.exists("models"):
    os.makedirs("models")

pickle.dump(best_model, open("models/best_model.pkl", "wb"))
print("Best placement model saved:", best_name)

# Save placement model metrics
perf_df = pd.DataFrame(results)
perf_df.to_csv("model_performance.csv", index=False)
print("Placement metrics saved to model_performance.csv")

# ============================================================
# STEP 8 - Train salary model (only on placed students)
# ============================================================
placed_df = df[df["placement_status"] == 1].copy()

X_salary = scaler.transform(placed_df[features])
y_salary = placed_df["salary_lpa"]

# Split salary data as well
X_sal_train, X_sal_test, y_sal_train, y_sal_test = train_test_split(
    X_salary, y_salary,
    test_size=0.2,
    random_state=42
)

salary_model = RandomForestRegressor(n_estimators=100, random_state=42)
salary_model.fit(X_sal_train, y_sal_train)

# ============================================================
# STEP 9 - Evaluate salary model
# ============================================================
sal_preds = salary_model.predict(X_sal_test)

mse  = mean_squared_error(y_sal_test, sal_preds)
rmse = np.sqrt(mse)
mae  = mean_absolute_error(y_sal_test, sal_preds)
r2   = r2_score(y_sal_test, sal_preds)

print("Salary Model Results:")
print("  MSE  :", round(mse,  4))
print("  RMSE :", round(rmse, 4), "LPA")
print("  MAE  :", round(mae,  4))
print("  R2   :", round(r2,   4))

# Save salary metrics
salary_metrics = pd.DataFrame([{
    "Model": "Random Forest Regressor",
    "MSE"  : round(mse,  4),
    "RMSE" : round(rmse, 4),
    "MAE"  : round(mae,  4),
    "R2"   : round(r2,   4),
}])
salary_metrics.to_csv("salary_performance.csv", index=False)
print("Salary metrics saved to salary_performance.csv")

# ============================================================
# STEP 10 - Save salary model
# ============================================================
pickle.dump(salary_model, open("models/salary_model.pkl", "wb"))
print("Salary model saved")

print("")
print("===================================")
print("Training Complete!")
print("Best Model  :", best_name)
print("Accuracy    :", round(best_acc * 100, 2), "%")
print("Salary RMSE :", round(rmse, 2), "LPA")
print("===================================")
