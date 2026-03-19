import pandas as pd
import numpy as np
import sqlite3
import pickle
import os
from datetime import datetime

from sklearn.model_selection import train_test_split
from sklearn.preprocessing   import StandardScaler
from sklearn.linear_model    import LogisticRegression
from sklearn.tree            import DecisionTreeClassifier
from sklearn.ensemble        import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics         import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    mean_squared_error, mean_absolute_error, r2_score
)

print("Starting Retraining...")

# ============================================================
# STEP 1 - Load existing data from database
# ============================================================
conn = sqlite3.connect("placement.db")
df   = pd.read_sql("SELECT * FROM student_data", conn)
conn.close()
print("Loaded from database. Total rows:", len(df))

# ============================================================
# STEP 2 - Check if new data file exists and add it
# If you have new student data, save it as "new_data.csv"
# and place it in the same folder. This script will
# automatically add it to the database and retrain.
# ============================================================
if os.path.exists("new_data.csv"):
    new_df = pd.read_csv("new_data.csv")
    print("New data found! Rows:", len(new_df))

    # Give new students IDs that continue from existing ones
    max_id        = df["Student_ID"].max()
    new_df["Student_ID"] = range(int(max_id) + 1, int(max_id) + 1 + len(new_df))

    # Combine old + new data
    df = pd.concat([df, new_df], ignore_index=True)
    print("Combined total rows:", len(df))

    # Save updated data back to database
    conn = sqlite3.connect("placement.db")
    df.to_sql("student_data", conn, if_exists="replace", index=False)
    conn.close()
    print("Updated database saved")
else:
    print("No new_data.csv found. Retraining on existing data only.")

# ============================================================
# STEP 3 - Clean and prepare data
# ============================================================
df["extracurricular_involvement"] = df["extracurricular_involvement"].fillna("Unknown")

df["placement_status"] = df["placement_status"].map({
    "Placed"    : 1,
    "Not Placed": 0
})

# Handle case where data was already numeric (after first retrain)
df["placement_status"] = pd.to_numeric(df["placement_status"], errors="coerce")
df = df.dropna(subset=["placement_status"])
df["placement_status"] = df["placement_status"].astype(int)

print("Placed   :", df["placement_status"].sum())
print("Not placed:", (df["placement_status"] == 0).sum())

# ============================================================
# STEP 4 - Features
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
# STEP 5 - Split first, then scale (no data leakage)
# ============================================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

scaler     = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)

# ============================================================
# STEP 6 - Check previous best accuracy
# ============================================================
prev_best_acc = 0
if os.path.exists("model_performance.csv"):
    prev_perf     = pd.read_csv("model_performance.csv")
    prev_best_acc = prev_perf["Accuracy"].max()
    print("Previous best accuracy:", round(prev_best_acc * 100, 2), "%")
else:
    print("No previous model found. This is the first run.")

# ============================================================
# STEP 7 - Train models
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
# STEP 8 - Save updated models
# ============================================================
if not os.path.exists("models"):
    os.makedirs("models")

pickle.dump(best_model, open("models/best_model.pkl", "wb"))
pickle.dump(scaler,     open("scaler.pkl",             "wb"))
print("New best model saved:", best_name)

# Update model_performance.csv
perf_df = pd.DataFrame(results)
perf_df.to_csv("model_performance.csv", index=False)

# ============================================================
# STEP 9 - Retrain salary model
# ============================================================
placed_df = df[df["placement_status"] == 1].copy()
X_salary  = scaler.transform(placed_df[features])
y_salary  = placed_df["salary_lpa"]

X_sal_train, X_sal_test, y_sal_train, y_sal_test = train_test_split(
    X_salary, y_salary, test_size=0.2, random_state=42
)

salary_model = RandomForestRegressor(n_estimators=100, random_state=42)
salary_model.fit(X_sal_train, y_sal_train)

sal_preds = salary_model.predict(X_sal_test)
mse  = mean_squared_error(y_sal_test, sal_preds)
rmse = np.sqrt(mse)
mae  = mean_absolute_error(y_sal_test, sal_preds)
r2   = r2_score(y_sal_test, sal_preds)

pickle.dump(salary_model, open("models/salary_model.pkl", "wb"))

salary_metrics = pd.DataFrame([{
    "Model": "Random Forest Regressor",
    "MSE"  : round(mse,  4),
    "RMSE" : round(rmse, 4),
    "MAE"  : round(mae,  4),
    "R2"   : round(r2,   4),
}])
salary_metrics.to_csv("salary_performance.csv", index=False)
print("Salary model retrained and saved")

# ============================================================
# STEP 10 - Save history log
# Every time you retrain, one row is added to this file
# so you can track how the model improved over time
# ============================================================
timestamp  = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

history_row = pd.DataFrame([{
    "Timestamp" : timestamp,
    "Best_Model": best_name,
    "Accuracy"  : round(best_acc, 4),
    "Total_Rows": len(df),
}])

if os.path.exists("model_history.csv"):
    old_history = pd.read_csv("model_history.csv")
    history     = pd.concat([old_history, history_row], ignore_index=True)
else:
    history = history_row

history.to_csv("model_history.csv", index=False)
print("History log updated in model_history.csv")

print("")
print("===================================")
print("Retraining Complete!")
print("Best Model :", best_name)
print("Accuracy   :", round(best_acc * 100, 2), "%")
print("Total Data :", len(df), "students")
print("===================================")
