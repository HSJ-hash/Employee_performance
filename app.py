# Import necessary libraries
import streamlit as st
import pandas as pd
import pickle

# Set up the Streamlit page
st.title("Employee Performance Prediction App")
st.write("This app uses a Random Forest model to predict employee performance based on various factors such as the number of workers, SMV, idle time, idle men, and tenure.")

# Load the saved model
model_filename = 'employee_performance_model.pkl'  # Ensure this file is in your working directory
with open(model_filename, 'rb') as file:
    loaded_model = pickle.load(file)

# Set predefined ranges and categories based on the original data structure
no_of_workers_range = (2, 89)  # Minimum and maximum values for 'No_of_workers'
smv_range = (2.9, 54.56)       # Minimum and maximum values for 'SMV'
idle_time_range = (0, 270)     # Minimum and maximum values for 'Idle_time'
idle_men_range = (0, 45)       # Minimum and maximum values for 'Idle_men'
tenure_options = [1, 2, 3, 4]  # Unique values for 'Tenure'
department_options = ['Legal and Compliance', 'Sales and Marketing', 'Product Management', 'Research and Development', 'Quality Assurance']  # Add departments from your dataset

# Sidebar input widgets for prediction
st.sidebar.header("Input Features for New Prediction")
no_of_workers = st.sidebar.slider('Number of Workers', no_of_workers_range[0], no_of_workers_range[1], 34)
smv = st.sidebar.slider('SMV', smv_range[0], smv_range[1], 15.15)
idle_time = st.sidebar.slider('Idle Time', idle_time_range[0], idle_time_range[1], 0)
idle_men = st.sidebar.slider('Idle Men', idle_men_range[0], idle_men_range[1], 0)
tenure = st.sidebar.selectbox('Tenure', tenure_options)
department = st.sidebar.selectbox('Department', department_options)  # Include Department selection

# Prepare the input data for prediction
input_data = pd.DataFrame({
    'No_of_workers': [no_of_workers],
    'SMV': [smv],
    'Idle_time': [idle_time],
    'Idle_men': [idle_men],
    'Tenure': [tenure],
    'Department': [department]  # Include Department in the data for model prediction
})

# Display the input data excluding the department column if necessary
st.subheader("Input Data for Prediction")
st.write(input_data.drop(columns=['Department']))

# Make prediction using the loaded model
prediction = loaded_model.predict(input_data)

# Display the prediction result
st.subheader("Predicted Performance Percentage")
st.write(f"Based on the input features, the predicted performance percentage is: **{prediction[0]:.2f}%**")

# Optional: Provide some explanation about the prediction process
st.info("The prediction is made using a Random Forest model that takes into account the provided features. Adjust the inputs on the left sidebar to see how they impact the performance percentage.")
