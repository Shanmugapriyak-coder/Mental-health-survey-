import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import joblib

from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
import pandas as pd

ohe_columns = joblib.load(r'C:\Users\MY Laptop\Desktop\guvi_class\mental health survey\ohe_columns.pkl')

class CustomPreprocessor(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.nominal_data = ['Gender','Working Professional or Student','Profession','Degree','Sleep Duration','Dietary Habits']
        self.allowed_profession = ['Teacher','Content Writer','Architect','Consultant','HR Manager','Pharmacist','Doctor','Business Analyst','Entrepreneur','Chemist']
        self.allowed_sleepduration = ['Less than 5 hours','7-8 hours','More than 8 hours','5-6 hours','3-4 hours']
        self.allowed_habits = ['Moderate','Unhealthy','Healthy']
        self.allowed_degree = ['Class 12','B.Ed','B.Arch','B.Com','B.Pharm','BCA','M.Ed','MCA','BBA','BSc']
    
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        df = X.copy()
        # Set academic values to 0 for professionals
        df.loc[df['Working Professional or Student'] == 'Working Professional', ['CGPA', 'Academic Pressure', 'Study Satisfaction']] = 0.0

        # Fill missing values
        for col in ['Profession', 'Degree', 'Dietary Habits']:
            df[col] = df[col].fillna(df[col].mode()[0])
        for col in ['Work Pressure', 'Academic Pressure', 'Job Satisfaction', 'Financial Stress', 'CGPA', 'Study Satisfaction']:
            df[col] = df[col].fillna(df[col].median())

        # Map uncommon categories to 'Others'
        df['Profession'] = df['Profession'].where(df['Profession'].isin(self.allowed_profession), 'Others')
        df['Sleep Duration'] = df['Sleep Duration'].where(df['Sleep Duration'].isin(self.allowed_sleepduration), 'Others')
        df['Dietary Habits'] = df['Dietary Habits'].where(df['Dietary Habits'].isin(self.allowed_habits), 'Others')
        df['Degree'] = df['Degree'].where(df['Degree'].isin(self.allowed_degree), 'Others')

        # Encode ordinal
        df['Family History of Mental Illness'] = df['Family History of Mental Illness'].replace({'No': 0, 'Yes': 1})
        df['Have you ever had suicidal thoughts ?'] = df['Have you ever had suicidal thoughts ?'].replace({'No': 0, 'Yes': 1})

        # One-hot encoding
        df = pd.get_dummies(df, columns=self.nominal_data, dtype='int')
        df = df.reindex(columns=ohe_columns, fill_value=0)
        # Handle outliers
        for col in ['Academic Pressure','CGPA','Study Satisfaction']:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower = Q1 - 1.5 * IQR
            upper = Q3 + 1.5 * IQR
            df[col] = np.clip(df[col], lower, upper)

        return df


# -------- Define the Model Class ----------
class DNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(DNN, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, hidden_size[0]),
            nn.ReLU(),
            nn.Linear(hidden_size[0], hidden_size[1]),
            nn.ReLU(),
            nn.Linear(hidden_size[1], hidden_size[2]),
            nn.ReLU(),
            nn.Linear(hidden_size[2], hidden_size[3]),
            nn.ReLU(),
            nn.Linear(hidden_size[3], output_size),
            nn.Sigmoid()
        )
        
    def forward(self, X):
        return self.layers(X)


# -------- Load Model & Preprocessor ----------
@st.cache_resource
def load_model():
    input_size = input_tensor.shape[1]
    hidden_size = [32,16,8,4]
    output_size = 1
    model = DNN(input_size, hidden_size, output_size)
    # model.load_state_dict(torch.load('model/mental_health_dnn.pth'))
    model.load_state_dict(torch.load(r'C:\Users\MY Laptop\Desktop\guvi_class\mental health survey\mental_health_dnn.pth'))
    model.eval()
    return model

@st.cache_resource
def load_pipeline():
    # return joblib.load('preprocessor.pkl')
    return joblib.load(r'C:\Users\MY Laptop\Desktop\guvi_class\mental health survey\preprocessor.pkl')




# -------- Streamlit UI ----------
st.title("🧠 Mental Health Prediction App")

# Example user input form
gender = st.selectbox("Gender", ["Male", "Female", "Others"])
Age = st.number_input("Age", min_value=1, max_value=100, step=1)
working_status = st.selectbox("Working Professional or Student", ["Working Professional", "Student"])
profession = st.selectbox("Profession", ["Teacher", "Content Writer", "Doctor","Student", "Others"])
degree = st.selectbox("Degree", ["BSc", "MCA", "B.Com", "Others"])
sleep = st.selectbox("Sleep Duration", ["5-6 hours", "7-8 hours", "Less than 5 hours", "Others"])
diet = st.selectbox("Dietary Habits", ["Healthy", "Moderate", "Unhealthy", "Others"])
cgpa = st.slider("CGPA", 0.0, 10.0, 7.5)
academic_pressure = st.slider("Academic Pressure", 0.0, 10.0, 5.0)
Work_Pressure=st.slider("Work Pressure", 0.0, 10.0, 5.0)
Financial_Stress=st.slider("Financial Stress", 0.0, 10.0, 5.0)
study_satisfaction = st.slider("Study Satisfaction", 0.0, 10.0, 5.0)
Job_Satisfaction = st.slider("Job Satisfaction", 0.0, 10.0, 5.0)
Work_Study_Hours = st.number_input("Work/Study Hours", min_value=1, max_value=12, step=1)
suicidal = st.radio("Have you ever had suicidal thoughts?", ["Yes", "No"])
family_history = st.radio("Family History of Mental Illness", ["Yes", "No"])
# Add other features as needed

# -------- Predict ----------
if st.button("Predict"):
    user_input = pd.DataFrame([{
        'Gender': gender,
        'Age':Age,
        'Working Professional or Student': working_status,
        'Profession': profession,
        'Degree': degree,
        'Sleep Duration': sleep,
        'Dietary Habits': diet,
        'CGPA': cgpa,
        'Academic Pressure': academic_pressure,
        'Work Pressure': Work_Pressure,
        'Study Satisfaction': study_satisfaction,
        'Financial Stress':Financial_Stress,
        'Job Satisfaction':Job_Satisfaction,
        'Work/Study Hours':Work_Study_Hours,
        'Have you ever had suicidal thoughts ?': suicidal,
        'Family History of Mental Illness': family_history
        # Include all required features
    }])
   
    preprocessor = load_pipeline()
    # Preprocess
    processed = preprocessor.transform(user_input)
    input_tensor = torch.tensor(processed.values, dtype=torch.float32)
    st.write(input_tensor.shape[1])
    model = load_model()
   
    with torch.no_grad():
        pred = model(input_tensor).item()
        label = "🚨 Depressed" if pred >= 0.5 else "✅ Not Depressed"
        st.markdown(f"### Prediction: {label}")
        st.markdown(f"**Confidence:** {pred:.2f}")

