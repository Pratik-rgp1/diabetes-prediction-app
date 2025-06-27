# Diabetes Prediction App

## Project Overview
This project builds a machine learning model to predict the likelihood of diabetes in patients based on various health and lifestyle factors. The project includes data preprocessing, handling class imbalance, training two classification models (Logistic Regression and Random Forest), evaluating their performance, and deploying the best model with an interactive web interface using Gradio.

---

## Dataset
- Source: [iammustafatz/diabetes-prediction-dataset on Kaggle](https://www.kaggle.com/datasets/iammustafatz/diabetes-prediction-dataset)
- Size: 100,000 records
- Features:
  - Gender
  - Age
  - Hypertension
  - Heart Disease
  - Smoking History
  - BMI
  - HbA1c Level
  - Blood Glucose Level
  - Target: Diabetes (0 = No, 1 = Yes)

---

## Data Preprocessing
- Dropped rows where gender was "Other" due to very low count.
- Converted categorical variables (`gender`, `smoking_history`) into numeric form using one-hot encoding.
- Scaled numerical features (`age`, `bmi`, `HbA1c_level`, `blood_glucose_level`) using StandardScaler.
- Split data into training (80%) and testing (20%) sets.
- Addressed class imbalance (only ~8.5% diabetic cases) using Random Over-Sampling on the training set.

---

## Modeling
### Logistic Regression
- Trained on the resampled training data.
- Evaluation on test set:
  - Accuracy: 88.9%
  - Precision: 43.9%
  - Recall: 88.9%
  - F1-score: 58.8%
- Observations:
  - High recall means most diabetic cases were correctly identified.
  - Low precision indicates many false positives.

### Random Forest Classifier
- Trained on the same resampled training data.
- Evaluation on test set:
  - Accuracy: 96.4%
  - Precision: 86.0%
  - Recall: 71.5%
  - F1-score: 78.1%
- Observations:
  - Significant improvement in precision and F1-score compared to Logistic Regression.
  - Better balance between detecting diabetes and minimizing false positives.

---

## Feature Importance (Random Forest)
- Most influential features:
  1. HbA1c Level
  2. Blood Glucose Level
  3. Age
  4. BMI
- Lesser impact from hypertension, heart disease, gender, and smoking history.

---

## Deployment
- Model saved as `random_forest_diabetes_model.joblib` inside `/models` folder.
- Developed an interactive web app using [Gradio](https://gradio.app/) for real-time diabetes risk prediction.
- User inputs:
  - Gender
  - Age
  - Hypertension status
  - Heart disease status
  - Smoking history
  - BMI
  - HbA1c level
  - Blood glucose level
- The app preprocesses inputs, applies the model, and outputs the prediction.

---

## How to Run the App Locally
1. Clone the repository.
2. Install required packages:

   ```bash
   pip install -r requirements.txt
## Run the Gradio app:

```bash
python app.py
#Open the local URL provided by Gradio in your browser to use the Diabetes Prediction app.
