import numpy as np
from PIL import Image
import streamlit as st
import pickle

# Load the model
def load_model():
    with open('heart_disease_pred_model.pkl', 'rb') as file:
        model = pickle.load(file)
    return model

# Predict heart disease
def predict_heart(input_data):
    model = load_model()
    input_data_as_numpy_array = np.asarray(input_data)
    input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)
    prediction = model.predict(input_data_reshaped)
    return prediction

# Main function
def main():
    # Set page configuration
    st.set_page_config(page_title="Heart Disease Prediction", page_icon="❤️", layout="wide")

    # Custom CSS for styling
    st.markdown("""
        <style>
            .stButton>button {
                background-color: #4CAF50;
                color: white;
                padding: 10px 24px;
                border: none;
                border-radius: 4px;
                cursor: pointer;
                font-size: 16px;
            }
            .stButton>button:hover {
                background-color: #45a049;
            }
            .stNumberInput>div>div>input {
                font-size: 16px;
            }
            .stSelectbox>div>div>select {
                font-size: 16px;
            }
            .stMarkdown {
                font-size: 18px;
            }
            
            .created-by {
                font-size: 20px;
                font-weight: bold;
                color: #4CAF50;
            }
        </style>
    """, unsafe_allow_html=True)

    # Title and description
    st.title("Heart Disease Prediction")
    st.markdown("This app predicts the likelihood of heart disease based on patient data. Please fill in the details below and click 'Predict'.")

    # Sidebar
    with st.sidebar:
        st.markdown('<p class="created-by">Created by Andrew O.A.</p>', unsafe_allow_html=True)
        
        # Load and display profile picture
        profile_pic = Image.open("prof.jpeg")  # Replace with your image file path
        st.image(profile_pic, caption="Andrew O.A.", use_container_width=True, output_format="JPEG")

        st.title("About")
        st.info("This app uses a machine learning model to predict the likelihood of heart disease.")
        st.markdown("[GitHub](https://github.com/Andrew-oduola) | [LinkedIn](https://linkedin.com/in/andrew-oduola-django-developer)")

    result_placeholder = st.empty()

    # Input fields
    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.number_input("Age", min_value=1, max_value=100, value=25, help="Enter the age of the patient")
        sex = st.selectbox("Sex", ["Male", "Female"], help="Select the gender of the patient")
        cp = st.selectbox("Chest Pain Type", ["Typical Angina", "Atypical Angina", "Non-anginal Pain", "Asymptomatic"],
                          help="Select the chest pain type of the patient")
        trestbps = st.number_input("Resting Blood Pressure", min_value=50, max_value=200, value=120,
                                   help="Enter the resting blood pressure of the patient")
        chol = st.number_input("Cholesterol", min_value=50, max_value=500, value=200,
                               help="Enter the cholesterol level of the patient")
        
    with col2:
        fbs = st.selectbox("Fasting Blood Sugar", ["No", "Yes"], help="Select if the patient has fasting blood sugar > 120 mg/dl")
        restecg = st.selectbox("Resting Electrocardiographic Results", ["Normal", "ST-T Wave Abnormality", "Left Ventricular Hypertrophy"],
                               help="Select the resting electrocardiographic results of the patient")
        thalach = st.number_input("Maximum Heart Rate Achieved", min_value=50, max_value=200, value=150,
                                   help="Enter the maximum heart rate achieved by the patient")
        exang = st.selectbox("Exercise Induced Angina", ["No", "Yes"], help="Select if the patient has exercise induced angina")
        
    with col3:
        oldpeak = st.number_input("ST Depression Induced by Exercise Relative to Rest", min_value=0.0, max_value=10.0, value=0.0,
                                  help="Enter the ST depression induced by exercise relative to rest")
        slope = st.selectbox("Slope of the Peak Exercise ST Segment", ["Upsloping", "Flat", "Downsloping"],
                             help="Select the slope of the peak exercise ST segment")
        ca = st.number_input("Number of Major Vessels Colored by Flourosopy", min_value=0, max_value=4, value=0,
                             help="Enter the number of major vessels colored by flourosopy")
        thal = st.selectbox("Thalassemia", ["Normal", "Fixed Defect", "Reversable Defect"], help="Select the thalassemia of the patient")

    # Convert input data to model format
    sex = 1 if sex == "Male" else 0
    cp = {"Typical Angina": 0, "Atypical Angina": 1, "Non-anginal Pain": 2, "Asymptomatic": 3}[cp] 
    fbs = 1 if fbs == "Yes" else 0
    restecg = {"Normal": 0, "ST-T Wave Abnormality": 1, "Left Ventricular Hypertrophy": 2}[restecg]
    exang = 1 if exang == "Yes" else 0
    slope = {"Upsloping": 0, "Flat": 1, "Downsloping": 2}[slope]
    thal = {"Normal": 0, "Fixed Defect": 1, "Reversable Defect": 2}[thal]

    input_data = [age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]

    # Prediction button
    if st.button("Predict"):
        prediction = predict_heart(input_data)
        if prediction == 0:
            st.success("The patient is **not likely** to have heart disease.")
            result_placeholder.success("The patient is **not likely** to have heart disease.")
        else:
            st.error("The patient is **likely** to have heart disease.")
            result_placeholder.error("The patient is **likely** to have heart disease.")
        
        st.markdown("**Note:** This is a simplified model and may not be accurate for all cases. Please consult with a healthcare professional for a more accurate diagnosis.")

if __name__ == "__main__":
    main()