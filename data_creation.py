import pandas as pd

# Path to the original dataset
original_dataset_path = r"C:\Users\amals\Downloads\fraudTest.csv"

# Load the original dataset
original_data = pd.read_csv(original_dataset_path)

# Extract 5 non-fraudulent and 5 fraudulent transactions
non_fraudulent_sample = original_data[original_data['is_fraud'] == 0].head(5)
fraudulent_sample = original_data[original_data['is_fraud'] == 1].head(5)

# Concatenate the samples
sample_data = pd.concat([non_fraudulent_sample, fraudulent_sample], ignore_index=True)

# Save the sample data to a new CSV file
sample_data.to_csv("sample_data.csv", index=False)

# Display the sample data
print("Sample Data:")
print(sample_data)
