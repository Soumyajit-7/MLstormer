# MLstormer: Automated Machine Learning and Data Profiling

**MLstormer** is a Streamlit application that provides an intuitive interface for performing data analysis and machine learning tasks. It allows users to upload datasets, explore data, train machine learning models, and make predictions. The application leverages the PyCaret library for machine learning and the YData Profiling library for data profiling.

## Features

- **Upload Dataset:** Upload your dataset for analysis and machine learning.
- **Data Profiling:** Perform exploratory data analysis using automated profiling.
- **Machine Learning:** Train and evaluate regression or classification models.
- **Download Model:** Save and download the best-performing machine learning model.
- **Make Predictions:** Use the trained model to make predictions on new data.

## Installation

1. **Clone the Repository**

   ```bash
   git clone https://github.com/your-repo/MLstormer.git
   cd MLstormer
2. **Install Requirements**
   Ensure you have requirements.txt in the project directory, then install dependencies:
   ```bash
   pip install -r requirements.txt
3. **Run the Streamlit App**

   ```bash
   streamlit run app.py

## Features

- **Upload Dataset:** Upload your dataset for analysis and machine learning.
- **Data Profiling:** Perform exploratory data analysis using automated profiling.
- **Machine Learning:** Train and evaluate regression or classification models.
- **Download Model:** Save and download the best-performing machine learning model.
- **Make Predictions:** Use the trained model to make predictions on new data.

## Functionality Overview

### Upload

- **Purpose:** Allows users to upload a CSV file containing the dataset.
- **Details:** Uploaded datasets are saved locally as `sourcedata.csv` and displayed in a table.

### Profiling

- **Purpose:** Provides exploratory data analysis (EDA) using automated profiling.
- **Details:** Generates a profile report of the dataset using the YData Profiling library and displays it in the Streamlit app.

### Machine Learning

- **Purpose:** Trains and evaluates machine learning models on the dataset.
- **Details:**
  - **Regression:** Uses PyCaret's regression module to set up, compare, and save the best regression model.
  - **Classification:** Uses PyCaret's classification module to set up, compare, and save the best classification model.
  - **Clustering:** (Commented Out) Optionally supports clustering models with PyCaret's clustering module.

### Download

- **Purpose:** Allows users to download the best-performing machine learning model.
- **Details:** The model is saved as `Best_Model.pkl` and can be downloaded through a Streamlit button.

### Predict

- **Purpose:** Makes predictions using the trained model.
- **Details:** Users provide input values for the model (excluding the target variable), and the app displays the predicted result.

## Code Explanation

### Upload Section

- **`file_uploader`**: Provides an interface for users to upload their dataset.
- **`pd.read_csv`**: Reads the uploaded CSV file into a DataFrame and saves it as `sourcedata.csv`.

### Profiling Section

- **`ProfileReport`**: Creates a comprehensive profile report of the dataset.
- **`st_profile_report`**: Displays the profile report in the Streamlit app.

### ML Section

- **`setup`**: Initializes the PyCaret environment for regression or classification.
- **`compare_models`**: Compares different models and selects the best one.
- **`save_model`**: Saves the best model to disk.
- **`pull`**: Retrieves and displays model comparison results.

### Download Section

- **`download_button`**: Allows users to download the saved model file.

### Predict Section

- **`cla.load_model`**: Loads the saved classification model (or adjust for regression).
- **`number_input` and `selectbox`**: Collects input values for predictions.
- **`predict`**: Makes predictions using the loaded model and displays the result.


   
