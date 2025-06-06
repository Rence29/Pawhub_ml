import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import os

# --- Configuration ---
PROCESSED_DATA_FILE = 'dog_adoption_preprocessed.csv'
MODEL_FILE = 'random_forest_dog_matcher_model_tuned.pkl' # Changed model file name to reflect tuning
ENCODED_FEATURES_FILE = 'encoded_features.txt' # To load the order of features if needed

# --- 1. Load Preprocessed Data ---
print(f"Loading preprocessed data from {PROCESSED_DATA_FILE}...")
if not os.path.exists(PROCESSED_DATA_FILE):
    print(f"Error: {PROCESSED_DATA_FILE} not found. Please run the preprocessing script first.")
    exit()

try:
    df = pd.read_csv(PROCESSED_DATA_FILE)
    print("Preprocessed data loaded successfully.")
    print("\nPreprocessed Data Head:")
    print(df.head())
except Exception as e:
    print(f"Error loading preprocessed data: {e}")
    exit()

# --- 2. Define Features (X) and Target (y) ---
# Target variable is 'Match_Outcome', features are all other columns
features = [col for col in df.columns if col != 'Match_Outcome']
X = df[features]
y = df['Match_Outcome']

print(f"\nFeatures (X) shape: {X.shape}")
print(f"Target (y) shape: {y.shape}")
print(f"Number of features: {len(features)}")
print(f"List of features: {features}")


# --- 3. Split Data into Training and Testing Sets ---
# 80% train, 20% test, stratify to maintain target distribution
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\nTraining set size: {len(X_train)} samples")
print(f"Testing set size: {len(X_test)} samples")

# --- 4. Initialize and Train the Random Forest Classifier with Tuned Hyperparameters ---
print("\nInitializing Random Forest Classifier with specified hyperparameters...")
tuned_parameters = {
    'n_estimators': 363,
    'max_depth': 45,
    'min_samples_leaf': 21,
    'min_samples_split': 20
}

# Note: Removed 'class_weight='balanced'' as specific min_samples_leaf/split are provided
model = RandomForestClassifier(
    **tuned_parameters,
    random_state=42 # Ensure reproducibility
)

print(f"Training Random Forest Classifier with parameters: {tuned_parameters}...")
model.fit(X_train, y_train)
print("Model training complete.")

# --- 5. Evaluate the Model ---
print("\nEvaluating model performance on the test set...")
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
report = classification_report(
    y_test, y_pred,
    target_names=['Bad Match', 'Good Match'],
    zero_division=0
)

print(f"Accuracy: {accuracy:.4f}")
print("\nConfusion Matrix:")
print(conf_matrix)
print("\nClassification Report:")
print(report)

if accuracy < 0.75:
    print("\n--- WARNING ---")
    print("Model accuracy is relatively low. Consider:")
    print("1. Re-evaluating the chosen hyperparameters.")
    print("2. Investigating feature importance to refine the dataset.")
    print("3. Exploring other machine learning algorithms or ensemble methods.")
    print("-----------------")

# --- 6. Save the Trained Model ---
joblib.dump(model, MODEL_FILE)
print(f"\nModel saved successfully to {MODEL_FILE}")

print("\nTraining script finished.")
