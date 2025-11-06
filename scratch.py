import streamlit as st
import pickle
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

with open('decision_tree_fetus.pickle', 'rb') as f:
    dt_model = pickle.load(f)
with open('random_forest_fetus.pickle', 'rb') as f:
    rf_model = pickle.load(f)
with open('adaboost_fetus.pickle', 'rb') as f:
    ab_model = pickle.load(f)
with open('voting_fetus.pickle', 'rb') as f:
    voting_model = pickle.load(f)

feature_names = [
    'baseline value', 'accelerations', 'fetal_movement',
    'uterine_contractions', 'light_decelerations', 'severe_decelerations',
    'prolongued_decelerations', 'abnormal_short_term_variability',
    'mean_value_of_short_term_variability', 'percentage_of_time_with_abnormal_long_term_variability',
    'mean_value_of_long_term_variability', 'histogram_width',
    'histogram_min', 'histogram_max', 'histogram_number_of_peaks',
    'histogram_number_of_zeroes', 'histogram_mode', 'histogram_mean',
    'histogram_median', 'histogram_variance', 'histogram_tendency'
]

st.title('Fetal Health Classification: A Machine Learning App')
st.image('fetal_health_image.gif', width=600)
st.write("Utilize our advanced Machine Learning application to predict fetal health classifications.")

st.sidebar.header("Fetal Health Features Input")
fetus_file = st.sidebar.file_uploader("Upload your data", type=['csv'])
st.sidebar.warning("⚠️ Ensure your data strictly follows the format outlined below.")
model_name = st.sidebar.radio("Choose Models for Prediction", ('Random Forest', 'Decision Tree', 'AdaBoost', 'Soft Voting'))
st.sidebar.info(f"You selected: {model_name}")
fetus_df = pd.read_csv('fetal_health.csv')
st.sidebar.dataframe(fetus_df.head())

if fetus_file is not None:
    input_df = pd.read_csv(fetus_file)
    st.success("✅ CSV file uploaded successfully!")
    st.subheader("Uploaded CSV Data")
    st.dataframe(input_df)
    for col in feature_names:
        if col not in input_df.columns:
            input_df[col] = 0
    input_df = input_df[feature_names]
    if model_name == "Random Forest":
        model = rf_model
    elif model_name == "Decision Tree":
        model = dt_model
    elif model_name == "AdaBoost":
        model = ab_model
    else:
        model = voting_model
    preds = model.predict(input_df)
    probs = model.predict_proba(input_df)
    class_mapping = {1: "Normal", 2: "Suspect", 3: "Pathological"}
    pred_labels = [class_mapping[p] for p in preds]
    input_df['Predicted Class'] = pred_labels
    input_df['Prediction Probability'] = (probs.max(axis=1) * 100).round(2)
    def color_class(val):
        if val == "Normal":
            color = 'lime'
        elif val == "Suspect":
            color = 'yellow'
        else:
            color = 'orange'
        return f'background-color: {color}'
    st.header(f"Predicting Fetal Health Class Using {model_name} Model")
    st.dataframe(input_df.style.applymap(color_class, subset=['Predicted Class']))
    st.subheader("Model Performance and Insights")
    tab1, tab2, tab3 = st.tabs(["Confusion Matrix", "Classification Report", "Feature Importance"])
    cm_files = {
        "Random Forest": "confusion_mat_rf.svg",
        "Decision Tree": "confusion_mat_dt.svg",
        "AdaBoost": "confusion_mat_ab.svg",
        "Soft Voting": "confusion_mat_sv.svg"
    }
    cr_files = {
        "Random Forest": "classification_report_rf.svg",
        "Decision Tree": "classification_report_dt.svg",
        "AdaBoost": "classification_report_ab.svg",
        "Soft Voting": "classification_report_sv.svg"
    }
    fi_files = {
        "Random Forest": "feature_importance_rf.svg",
        "Decision Tree": "feature_importance_dt.svg",
        "AdaBoost": "feature_importance_ab.svg",
        "Soft Voting": "feature_importance_sv.svg"
    }
    with tab1:
        st.image(cm_files[model_name])
    with tab2:
        st.image(cr_files[model_name])
    with tab3:
        st.image(fi_files[model_name])
else:
    st.info("Please upload data to proceed.")
