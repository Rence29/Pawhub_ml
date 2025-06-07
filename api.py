from flask import Flask, request, jsonify
import joblib
import pandas as pd
import os
import re # Although not strictly needed for integer age, keeping for robustness if other string parsing is added later.

app = Flask(__name__)

# --- Configuration ---
# Ensure this matches the output of your training script for the tuned model
MODEL_FILE = 'random_forest_dog_matcher_model_tuned.pkl'
ENCODED_FEATURES_FILE = 'encoded_features.txt'
PORT = 5000 # Choose an available port (make sure it's not in use by other applications)

# --- Load the Trained Model and Feature List ---
# These are loaded only once when the API starts, for efficiency.
try:
    model = joblib.load(MODEL_FILE)
    print(f"Model '{MODEL_FILE}' loaded successfully.")

    with open(ENCODED_FEATURES_FILE, 'r') as f:
        expected_features = [line.strip() for line in f]
    print(f"Loaded {len(expected_features)} expected features from {ENCODED_FEATURES_FILE}.")

except FileNotFoundError:
    print(f"Error: Required file not found. Make sure '{MODEL_FILE}' and '{ENCODED_FEATURES_FILE}' exist.")
    print("Please run preprocess_data.py and train_model.py first.")
    exit()
except Exception as e:
    print(f"Error loading model or features: {e}")
    exit()

# Function to preprocess incoming data from the API request
def preprocess_input(data):
    """
    Applies the same preprocessing steps to a single input data point
    as were applied to the training data.

    Expected input 'data' dictionary keys (matching original CSV column names):
    - 'dog_age': integer (e.g., 2, 5)
    - 'house_type': string (e.g., 'Medium', 'Large', 'Small')
    - 'family_composition': string (e.g., 'With kids', 'No kids')
    - 'lifestyle': string (e.g., 'Sedentary', 'Active')
    - 'pet_experience': string (e.g., 'Yes', 'No')
    - 'dog_size': string (e.g., 'Small', 'Medium', 'Large')
    - 'dog_behavior': string (e.g., 'Aggressive', 'Energetic', 'Calm')
    - 'health_condition': string (e.g., 'Healthy')
    """
    processed_data_dict = {}

    # Initialize all expected features to 0 to ensure consistency with training data
    for feature_name in expected_features:
        processed_data_dict[feature_name] = 0

    # --- Process 'dog_age' ---
    # Expected as a numerical value (integer or float) directly
    dog_age = data.get('dog_age')
    if dog_age is not None:
        try:
            processed_data_dict['dog_age'] = float(dog_age)
        except ValueError:
            print(f"Warning: Could not convert 'dog_age' value '{dog_age}' to float. Setting to 0.")
            processed_data_dict['dog_age'] = 0 # Default to 0 or handle as needed
    else:
        print("Warning: 'dog_age' key missing in input data. Setting to 0.")
        processed_data_dict['dog_age'] = 0


    # --- Process Categorical Features (One-Hot Encoding) ---
    # These are the original column names expected in the incoming JSON request
    api_categorical_cols = [
        'house_type', 'family_composition', 'lifestyle', 'pet_experience',
        'dog_size', 'dog_behavior', 'health_condition'
    ]

    for col in api_categorical_cols:
        value = data.get(col)
        if value:
            # Construct the one-hot encoded column name as created by pd.get_dummies
            # pd.get_dummies replaces spaces with underscores by default
            cleaned_value = value.replace(' ', '_')
            encoded_col_name = f"{col}_{cleaned_value}"

            # Set the corresponding one-hot encoded feature to 1 if it exists in expected features
            if encoded_col_name in expected_features:
                processed_data_dict[encoded_col_name] = 1
            else:
                print(f"Warning: Encoded feature '{encoded_col_name}' (from '{col}': '{value}') not found in expected features. This column will remain 0.")
        else:
            print(f"Warning: Categorical feature '{col}' missing in input data.")


    # Create a DataFrame from the processed data_dict, ensuring correct column order.
    # Any features not present in the input (and thus not explicitly set to 1)
    # will retain their initialized 0 value.
    input_df = pd.DataFrame([processed_data_dict], columns=expected_features)

    return input_df

@app.route('/predict_match', methods=['POST'])
def predict_match():
    """
    Receives adopter and dog data (combined in one JSON),
    preprocesses it, makes a prediction using the loaded model,
    and returns "Good Match" or "Bad Match".

    Expected JSON structure for POST request:
    {
        "dog_age": 3,
        "house_type": "Medium",
        "family_composition": "With Kids",
        "lifestyle": "Sedentary",
        "pet_experience": "Yes",
        "dog_size": "Small",
        "dog_behavior": "Aggressive",
        "health_condition": "Healthy"
    }
    """
    if not request.json:
        return jsonify({'error': 'Request must be JSON'}), 400

    raw_data = request.json
    print(f"Received raw prediction request data: {raw_data}")

    # Preprocess the incoming data
    try:
        input_for_prediction = preprocess_input(raw_data)
        # Optional: uncomment for debugging
        # print("Preprocessed input for prediction:")
        # print(input_for_prediction)

    except Exception as e:
        print(f"Preprocessing error: {e}")
        return jsonify({'error': f'Error during input preprocessing: {e}'}), 400

    # Crucial check: Ensure the preprocessed DataFrame has the exact same columns and order as expected by the model
    # This prevents errors if input data is malformed or if feature lists differ.
    if list(input_for_prediction.columns) != expected_features:
        print(f"Mismatched features. Expected: {expected_features}, Got: {list(input_for_prediction.columns)}")
        return jsonify({
            'error': 'Input features do not match expected model features. '
                     'Please check your input data keys and values against the expected format.'
        }), 400

    try:
        # Make the prediction (returns 0 for Bad Match, 1 for Good Match)
        prediction_int = model.predict(input_for_prediction)[0]

        # Map the integer prediction to the desired string output
        match_result = 'Good Match' if prediction_int == 1 else 'Bad Match'

        # Return only the match result
        return jsonify({
            'match_result': match_result,
            'message': 'Prediction successful'
        }), 200

    except Exception as e:
        print(f"Prediction error: {e}")
        return jsonify({'error': f'An error occurred during prediction: {e}'}), 500

@app.route('/health', methods=['GET'])
def health_check():
    """Simple health check endpoint to confirm API is running."""
    status = 'healthy'
    model_loaded = False
    if 'model' in globals() and model is not None:
        model_loaded = True
    else:
        status = 'degraded - model not loaded'
    return jsonify({'status': status, 'model_loaded': model_loaded, 'port': PORT}), 200


if __name__ == '__main__':
    print(f"Starting Flask API on http://127.0.0.1:{PORT}")
    print("Keep this terminal open for the API to run.")
    # Set debug=True for development to auto-reload on code changes
    # Set host='0.0.0.0' to make it accessible from other devices on your network
    app.run(debug=True, port=PORT, host='0.0.0.0')
