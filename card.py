import pandas as pd
import datetime as dt
import numpy as np
import category_encoders as ce
from lightgbm import LGBMClassifier
import joblib

# Load the trained model
model_filename = 'lgbm_model.joblib'
lgbm_model = joblib.load(model_filename)

# Example Usage for Train Data
columns_to_encode = ["category", "state", "city", "job"]
intervals = [600, 1200, 1800, 2400, 3000, 3600]
gender_mapping = {"F": 0, "M": 1}

def apply_woe(data, columns, target_col):
    woe = ce.WOEEncoder()

    for col in columns:
        X = data[col]
        y = data[target_col]

        new_col_name = f"{col}_WOE"
        data[new_col_name] = woe.fit_transform(X, y)

    return data

def classify_frequency(freq, intervals):
    for i, c in enumerate(intervals):
        if freq <= c:
            return i

def preprocess_new_data(data, columns_to_encode, gender_mapping, intervals):
    data['age'] = dt.date.today().year - pd.to_datetime(data['dob']).dt.year
    data['hour'] = pd.to_datetime(data['trans_date_trans_time']).dt.hour
    data['month'] = pd.to_datetime(data['trans_date_trans_time']).dt.month

    data.drop(columns=["merchant", "first", "last", "street",
                       "unix_time", "trans_num", "Unnamed: 0"], inplace=True)

    data["amt_log"] = np.log1p(data["amt"])

    data = apply_woe(data, columns_to_encode, "is_fraud")
    
    data["gender_binary"] = data["gender"].map(gender_mapping)

    freq_enc = (data.groupby("cc_num").size())
    freq_enc.sort_values(ascending=True)
    data["cc_num_frequency"] = data["cc_num"].apply(lambda x: freq_enc[x])
    data["cc_num_frequency_classification"] = data["cc_num_frequency"].apply(lambda x: classify_frequency(x, intervals))

    data.drop(columns=["trans_date_trans_time",
                       "city", "state", "category", "gender", "dob", "job", "cc_num", "amt",
                       "gender_binary", "state_WOE", "zip", "long", "lat",
                       "city_pop", "month", "cc_num_frequency_classification", "merch_long"], inplace=True)

    return data


def predict_fraud(data, model):
    # Assuming data is a DataFrame with the same structure as the training data
    X = data.drop(columns=["is_fraud"])
    y_pred = model.predict(X)
    return y_pred


# Example usage:
# Load the new data
new_data_path = 'sample_data.csv'
new_data = pd.read_csv(new_data_path)

# Preprocess the new data
new_data_processed = preprocess_new_data(new_data, columns_to_encode, gender_mapping, intervals)


# Make predictions
predictions = predict_fraud(new_data_processed, lgbm_model)

# Display the predictions
print("Predictions:")
print(predictions)


def predict_fraud_prob(data, model):
    # Assuming data is a DataFrame with the same structure as the training data
    X = data.drop(columns=["is_fraud"])
    y_pred_prob = model.predict_proba(X)[:, 1]
    return y_pred_prob

# Example usage:
# Load the new data
new_data_path = 'sample_data.csv'
new_data = pd.read_csv(new_data_path)

# Preprocess the new data
new_data_processed = preprocess_new_data(new_data, columns_to_encode, gender_mapping, intervals)

# Make predictions and get predicted probabilities
predicted_probabilities = predict_fraud_prob(new_data_processed, lgbm_model)

# Display the predictions, actual labels, and predicted probabilities
print("Predictions:")
print(predicted_probabilities)
