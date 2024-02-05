import streamlit as st 
import pandas as pd
import os
from ydata_profiling import ProfileReport
from streamlit_pandas_profiling import st_profile_report
#from pycaret.classification import setup, compare_models, pull, save_model
import pycaret.clustering as clu 
import pycaret.regression as reg
import pycaret.classification as cla
with st.sidebar:
    st.image("https://builtin.com/sites/www.builtin.com/files/styles/ckeditor_optimize/public/inline-images/machine-learning-pillar-page-overview.jpeg")
    st.title("MLstormer")
    choice=st.radio("Navigation", ["Upload", "Profiling", "ML", "Download","Predict"])
    st.info("This application Analyzes your data and does Machine Learning on it Automatically")

if os.path.exists("sourcedata.csv"):
    df=pd.read_csv("sourcedata.csv", index_col=None)

if choice == "Upload":
    st.title("Upload Your Dataset for Modelling")
    file = st.file_uploader("Upload Your Dataset")
    if file:
        df = pd.read_csv(file)
        df.to_csv("sourcedata.csv", index=None)
        st.dataframe(df)

if choice == "Profiling": 
    st.title("Exploratory Data Analysis")
    profile_df=ProfileReport(df)
    st_profile_report(profile_df)


if choice == "ML":
    
    st.title("It is Happening!!! I mean Machine Learning.......")
    choice1=st.radio("Select your Machine Learning Model", ["Regression","Classification"])
    tar=st.selectbox("Select your Target variable", df.columns)
    
    if choice1 == "Regression":
        if st.button("Start Machine Learning:"):
            reg.setup(df, target=tar, remove_outliers = True) 
            setup_df = reg.pull()
            st.info("ML Experiment Settings")
            st.dataframe(setup_df)
            best_model = reg.compare_models()
            compare_df = reg.pull()
            st.info("Best ML Model")
            st.dataframe(compare_df)
            best_model
            reg.save_model(best_model,"Best_Model")
    if choice1 == "Classification":
        if st.button("Start Machine Learning:"):
            cla.setup(df, target=tar, remove_outliers = True) 
            setup_df = cla.pull()
            st.info("ML Experiment Settings")
            st.dataframe(setup_df)
            best_model = cla.compare_models()
            compare_df = cla.pull()
            st.info("Best ML Model")
            st.dataframe(compare_df)
            best_model
            cla.save_model(best_model,"Best_Model")

    # if choice1 == "Clustering":
    #     if st.button("Start Machine Learning:"):
    #         clu.setup(df,) 
    #         setup_df = clu.pull()
    #         st.info("ML Experiment Settings")
    #         st.dataframe(setup_df)
    #         best_model = clu.compare_models()
    #         compare_df = clu.pull()
    #         st.info("Best ML Model")
    #         st.dataframe(compare_df)
    #         best_model
    #         clu.save_model(best_model,"Best_Model")


if choice == "Download":
    with open("Best_Model.pkl", "rb") as f:
        st.download_button("Download the best Model", f,"trained_model.pkl")

if choice == "Predict":
    st.title("Make Predictions")

    # Load the saved model
    loaded_model = cla.load_model("Best_Model")  # Assuming it's a regression model, adjust if it's a classification model

    # Get input values from the user
    input_values = {}
    for column in df.columns:
        if df[column].dtype == 'object':
            input_values[column] = st.selectbox(f"Select value for {column}", df[column].unique())
        else:
            input_values[column] = st.number_input(f"Enter value for {column}")

    # Exclude the target column from the input features
    target_column = st.selectbox("Select the target variable", df.columns)
    input_data = pd.DataFrame([input_values]).drop(columns=[target_column])

    # Make predictions
    prediction = loaded_model.predict(input_data)

    st.success(f"The predicted target value is: {prediction[0]}")
