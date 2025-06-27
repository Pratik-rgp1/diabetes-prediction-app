import gradio as gr
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler

# Load the trained RandomForest model
model = joblib.load('./models/random_forest_diabetes_model.joblib')

# Create a dummy dataset to fit the scaler (based on training structure)
dummy_data = pd.DataFrame({
    'age': [40.0],
    'bmi': [25.0],
    'HbA1c_level': [5.0],
    'blood_glucose_level': [100]
})
numerical_cols = ['age', 'bmi', 'HbA1c_level', 'blood_glucose_level']
scaler = StandardScaler()
scaler.fit(dummy_data)  # Note: Replace this with the actual scaler if saved separately

# Use the feature columns from training (update if needed)
X_train_columns = [
    'age', 'hypertension', 'heart_disease', 'bmi', 'HbA1c_level', 'blood_glucose_level',
    'gender_Male', 'smoking_history_current', 'smoking_history_ever',
    'smoking_history_former', 'smoking_history_never', 'smoking_history_not current'
]

# Prediction function
def predict_diabetes(gender, age, hypertension, heart_disease, smoking_history, bmi, hba1c_level, blood_glucose_level):
    new_data = pd.DataFrame({
        'gender': [gender],
        'age': [float(age)],
        'hypertension': [int(hypertension)],
        'heart_disease': [int(heart_disease)],
        'smoking_history': [smoking_history],
        'bmi': [float(bmi)],
        'HbA1c_level': [float(hba1c_level)],
        'blood_glucose_level': [int(blood_glucose_level)]
    })

    # One-hot encode categorical features
    categorical_cols = ['gender', 'smoking_history']
    new_data_encoded = pd.get_dummies(new_data, columns=categorical_cols, drop_first=True)

    # Add missing columns with 0 and align order
    for col in X_train_columns:
        if col not in new_data_encoded.columns:
            new_data_encoded[col] = 0
    new_data_encoded = new_data_encoded[X_train_columns]

    # Scale numerical features
    new_data_encoded[numerical_cols] = scaler.transform(new_data_encoded[numerical_cols])

    # Predict using the trained model
    prediction = model.predict(new_data_encoded)
    return "Prediction: Diabetes" if prediction[0] == 1 else "Prediction: No Diabetes"

# Gradio Interface
interface = gr.Interface(
    fn=predict_diabetes,
    inputs=[
        gr.Radio(['Female', 'Male'], label="Gender"),
        gr.Slider(0.08, 80.0, step=1, label="Age"),
        gr.Radio([0, 1], label="Hypertension (0=No, 1=Yes)"),
        gr.Radio([0, 1], label="Heart Disease (0=No, 1=Yes)"),
        gr.Dropdown(['never', 'No Info', 'current', 'former', 'ever', 'not current'], label="Smoking History"),
        gr.Slider(10.01, 95.69, step=0.1, label="BMI"),
        gr.Slider(3.5, 9.0, step=0.1, label="HbA1c Level"),
        gr.Slider(80, 300, step=1, label="Blood Glucose Level")
    ],
    outputs="text",
    title="🧪 Diabetes Prediction App",
    description="Enter patient information to predict diabetes using a trained RandomForest model."
)

# Launch the app
if __name__ == "__main__":
    interface.launch()
