# Credit Card Fraud Detection

This repository contains code for a credit card fraud detection project. The project includes data preprocessing, model development, and implementation using Python and LightGBM.

## Contents

1. [Model Creation](Model_creation.ipynb): Jupyter notebook for data preprocessing, model development, and saving the trained model.
2. [Data Creation](data_creation.py): Python script to create a sample data CSV file.
3. [Credit Card Fraud Implementation](credit_card_fraud.py): Python script to implement the model using sample data.
4. [Fraud Test Data](fraudtest.csv): Test dataset for fraud detection.
5. [Fraud Train Data](fraudtrain.csv): Train dataset for fraud detection.
6. [Sample Data](sample_data.csv): Sample data CSV file generated using data_creation.py.
7. [Saved Model](lgbm_model.joblib): Pretrained LightGBM model saved using joblib.
8. [.gitignore](.gitignore): Specifies intentionally untracked files to ignore for version control.

## Usage

1. **Download and Unzip the Dataset:**
   - Download the dataset from [this link](https://www.kaggle.com/datasets/kartik2112/fraud-detection/data).
   - Unzip the downloaded file to obtain the `fraudtest.csv` and `fraudtrain.csv` files.

2. **Model Creation:**
   - Open and run the [Model_creation.ipynb](Model_creation.ipynb) notebook to preprocess data, develop the model, and save it.

3. **Data Creation:**
   - Execute [data_creation.py](data_creation.py) to generate a sample data CSV file.

4. **Credit Card Fraud Implementation:**
   - Utilize [credit_card_fraud.py](credit_card_fraud.py) to implement the model on your data.

## Dataset

- **Fraud Test Data:** [fraudtest.csv](fraudtest.csv)
- **Fraud Train Data:** [fraudtrain.csv](fraudtrain.csv)

## Requirements

- Python 3.x
- Jupyter Notebook

**Note:** After downloading and unzipping the dataset, place the `fraudtest.csv` and `fraudtrain.csv` files in the project folder before running the code files.
