# Placement & Salary Prediction System

This project is a machine learning based system designed to predict whether a student is likely to get placed during campus recruitment and to estimate the expected salary package based on their academic performance and skill profile.

It was developed as part of Hackathon 3, focusing on building a complete ML pipeline — from data processing and model training to deployment through an interactive dashboard.

---

## About the Project

The system uses student data such as CGPA, internships, projects, and skill ratings to:

- Predict placement outcome (Placed / Not Placed)
- Estimate expected salary (LPA) for placed students
- Provide data insights and model performance analytics
- Support model retraining when new student data becomes available

The goal is to create a simple decision-support tool that demonstrates how machine learning can be applied to real-world career outcome analysis.

---

## Project Structure

```
Hackathon3/
│
├── student_placement.csv
├── train.py
├── retrain.py
├── Dashboard.py
│
├── models/
│   ├── best_model.pkl
│   └── salary_model.pkl
│
├── scaler.pkl
├── placement.db
├── model_performance.csv
├── salary_performance.csv
├── model_history.csv
└── README.md
```

---

## How to Run the Project

**Step 1 — Install dependencies**

```bash
pip install streamlit pandas numpy scikit-learn matplotlib seaborn
```

**Step 2 — Train the models**

```bash
python train.py
```

**Step 3 — Launch the dashboard**

```bash
streamlit run Dashboard.py
```

**Step 4 — Retrain the model (optional)**

If new student data is available, save it as `new_data.csv` in the same folder and run:

```bash
python retrain.py
```

The system will automatically update the dataset, retrain models, and log performance history.

---

## Dataset Details

- Total records: 5000 students
- Key features: CGPA, internships, projects, coding skills, communication skills, aptitude
- Target variables: `placement_status` and `salary_lpa`

Only the most relevant features are used for prediction to keep the model simple and effective.

---

## Machine Learning Approach

**Placement Prediction**

Three classification models are trained and compared:

- Logistic Regression
- Decision Tree
- Random Forest

The best performing model is automatically selected based on evaluation metrics such as Accuracy, Precision, Recall, F1 Score and ROC-AUC.

**Salary Prediction**

A Random Forest Regressor is used to estimate expected salary for students predicted as placed. Performance is measured using RMSE, MAE and R² score.

---

## Prediction Logic

As a practical rule, students with CGPA below 3.5 are marked as Not Placed, since most companies set a minimum eligibility cutoff. This combines domain knowledge with machine learning predictions.

---

## Model Lifecycle

Whenever retraining is performed:

- New data is merged into the existing database
- All models are retrained
- The best model is updated
- Training history is stored for performance tracking

This simulates a continuous learning pipeline similar to real-world ML systems.

---

## Technologies Used

- Python
- Scikit-learn
- Pandas and NumPy
- SQLite
- Streamlit
- Matplotlib and Seaborn
- Pickle

---

## Dashboard Highlights

- Placement analytics and visual insights
- Model accuracy comparison
- Retraining history tracking
- Real-time placement probability prediction
- Expected salary estimation

---

## Author

Kunal Mankar  
B.Tech Computer Science Engineering  
Hackathon 3 Submission
