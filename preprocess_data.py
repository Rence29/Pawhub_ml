import pandas as pd
import os

# --- Configuration ---
DATA_FILE = 'rf_dataset_updated.csv'  # your new CSV file
PROCESSED_DATA_FILE = 'dog_adoption_preprocessed.csv'
ENCODED_FEATURES_FILE = 'encoded_features.txt'

# --- 1. Load the dataset ---
print(f"Loading data from {DATA_FILE}...")
if not os.path.exists(DATA_FILE):
    print(f"Error: {DATA_FILE} not found. Please ensure your dataset is in the same directory.")
    exit()

df = pd.read_csv(DATA_FILE)
print("Data loaded successfully.")
print("\nOriginal Data Info:")
df.info()
print("\nFirst 5 rows of original data:")
print(df.head())

# --- 2. Preprocess 'match_category' (Target Variable) ---
print("\nPreprocessing 'match_category'...")
# Map 'Good Match' to 1 and 'Bad Match' to 0
df['Match_Outcome'] = df['match_category'].map({'Good Match': 1, 'Bad Match': 0})
df.drop('match_category', axis=1, inplace=True)
print(" 'match_category' converted to numerical 'Match_Outcome'.")

# --- 3. Handle Identifier Columns ---
print("\nDropping identifier columns 'adopter_id' and 'dog_id'...")
df.drop(['adopter_id', 'dog_id'], axis=1, inplace=True)
print("Identifier columns dropped.")

# --- 4. Handle Categorical Features (One-Hot Encoding) ---
print("\nPerforming One-Hot Encoding for categorical features...")
categorical_cols = [
    'house_type', 'family_composition', 'lifestyle', 'pet_experience',
    'dog_size', 'dog_behavior', 'health_condition'
]

df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=False)

print("One-Hot Encoding complete.")
print("\nFirst 5 rows of preprocessed data:")
print(df_encoded.head())
print("\nPreprocessed Data Info:")
df_encoded.info()

# --- 5. Save the preprocessed data and feature list ---
print(f"\nSaving preprocessed data to {PROCESSED_DATA_FILE}...")
df_encoded.to_csv(PROCESSED_DATA_FILE, index=False)
print("Preprocessed data saved.")

features_list = df_encoded.drop('Match_Outcome', axis=1).columns.tolist()
with open(ENCODED_FEATURES_FILE, 'w') as f:
    for item in features_list:
        f.write("%s\n" % item)
print(f"List of encoded features saved to {ENCODED_FEATURES_FILE}.")

print("\nData preprocessing complete. You can now proceed to train your model.")