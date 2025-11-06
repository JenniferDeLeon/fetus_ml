# App 1: Fetal Health Classification

# Import libraries
from sklearn.metrics import classification_report
import streamlit as st
import pickle
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Suppress warnings
import warnings
warnings.filterwarnings("ignore")

# Load Models 
with open('decision_tree_fetus.pickle', 'rb') as f:
    dt_model = pickle.load(f)

with open('random_forest_fetus.pickle', 'rb') as f:
    rf_model = pickle.load(f)

with open('adaboost_fetus.pickle', 'rb') as f:
    ab_model = pickle.load(f)

with open('voting_fetus.pickle', 'rb') as f:
    voting_model = pickle.load(f)

# Feature Names - Should match the order used in model training
feature_names = ['baseline value', 'accelerations', 'fetal_movement',
    'uterine_contractions', 'light_decelerations', 'severe_decelerations',
    'prolongued_decelerations', 'abnormal_short_term_variability',
    'mean_value_of_short_term_variability', 'percentage_of_time_with_abnormal_long_term_variability',
    'mean_value_of_long_term_variability', 'histogram_width',
    'histogram_min', 'histogram_max', 'histogram_number_of_peaks',
    'histogram_number_of_zeroes', 'histogram_mode', 'histogram_mean',
    'histogram_median', 'histogram_variance', 'histogram_tendency']

# Set up the title and description of the app
st.title('Fetal Health Classification: A Machine Learning App') 
st.image('fetal_health_image.gif', width = 600)
st.write("Utilize our advanced Machine Learning application to predict fetal health classifications.")

# Set up sidebar for user input
st.sidebar.header("Fetal Health Features Input")

# User upload 
fetus_file = st.sidebar.file_uploader("Upload your data", type=['csv'])

st.sidebar.warning("⚠️ Ensure your data strictly follows the format outlined below.")
fetus_df = pd.read_csv('fetal_health.csv')
st.sidebar.dataframe(fetus_df.head()) # This shows what the user's CSV should look like

# For user to choose model 
model_name = st.sidebar.radio("Choose Models for Prediction", ('Random Forest', 'Decision Tree', 'AdaBoost', 'Soft Voting'))

# Notification for selected model
st.sidebar.info(f"You selected: {model_name}")

# Only run predictions if user uploaded CSV
# I used ChatGPT here to help me understand what "is not None" means and how to implement it in the code
if fetus_file is not None: 
    input_df = pd.read_csv(fetus_file)
    st.success("✅ CSV file uploaded successfully!")

    # Class Mapping
    # I used ChatGPT here to double check the dictionary syntax
    class_mapping = {1: "Normal", 2: "Suspect", 3: "Pathological"}

    # Add missing columns with 0 and reorder to clean
    for col in feature_names:
        if col not in input_df.columns:
            input_df[col] = 0
    input_df = input_df[feature_names]

    # Select the learning model based on the user choice
    if model_name == "Random Forest":
        model = rf_model
    elif model_name == "Decision Tree":
        model = dt_model
    elif model_name == "AdaBoost":
        model = ab_model
    else:
        model = voting_model 

    # Make predictions using predict_proba
    preds = model.predict(input_df)
    probs = model.predict_proba(input_df)

    # Map numeric predictions to class labels
    # I used ChatGPT here to help me with the syntax for mapping numeric predictions to class labels
    pred_labels = [class_mapping[p] for p in preds]
    input_df['Predicted Fetal Health'] = pred_labels

    # Convert prediction probability to percentage (I used ChatGPT here to help me figure out the logic behind the prediction probability.)
    input_df['Prediction Probability (%)'] = (probs.max(axis=1) * 100).round(2)

    # I used ChatGPT here to help me figure out how to color code syntax for the Predicted Fetal Health column
    def color_class(val):
        if val == "Normal":
            color = 'lime'
        elif val == "Suspect":
            color = 'yellow'
        else:
            color = 'orange'
        return f'background-color: {color}'

    # Display predictions on main page
    st.header(f"Predicting Fetal Health Class Using {model_name} Model")
    st.dataframe(input_df.style.applymap(color_class, subset=['Predicted Fetal Health']))

    # Model insight tabs / additional features 
    st.subheader("Model Performance and Insights")
    tab1, tab2, tab3 = st.tabs(["Confusion Matrix", "Classification Report", "Feature Importance"])

    # Tab 1: Confusion Matrix
    with tab1:
        st.write("Confusion Matrix")
        if model_name =="Decision Tree":
            st.image('confusion_mat_dt.svg')
        elif model_name == "Random Forest":
            st.image('confusion_mat_rf.svg')
        elif model_name == "AdaBoost":
            st.image('confusion_mat_ab.svg')
        else:
            st.image('confusion_mat_voting.svg')

    # Tab 2: Classification Report
    # I used ChatGPT here. I could not figure out how to save the df as a .svg file.
    # Instead, I saved the classification report as a .csv file and read it here to display in Streamlit.
    with tab2:
        st.write("Classification Report")
        if model_name == "Decision Tree":
            df = pd.read_csv("classification_report_dt.csv", index_col=0)
            st.write(df.style.background_gradient(cmap='Greens'))
        elif model_name == "Random Forest":
            df = pd.read_csv("classification_report_rf.csv", index_col=0)
            st.write(df.style.background_gradient(cmap='Blues'))
        elif model_name == "AdaBoost":
            df = pd.read_csv("classification_report_ab.csv", index_col=0)
            st.write(df.style.background_gradient(cmap='Reds'))
        else:  # Soft Voting
            df = pd.read_csv("classification_report_voting.csv", index_col=0)
            st.write(df.style.background_gradient(cmap='Purples'))

    # Tab 3: Feature Importance
    with tab3:
        st.write("Feature Importance")
        if model_name =="Decision Tree":
            st.image('feature_imp_dt.svg')
        elif model_name == "Random Forest":
            st.image('feature_imp_rf.svg')
        elif model_name == "AdaBoost":
            st.image('feature_imp_ab.svg')
        else:
            st.image('feature_imp_voting.svg')
else: 
    st.info("ℹ️ Please upload data to proceed.") # The success message should only show if the user uploads a CSV, otherwise st.info