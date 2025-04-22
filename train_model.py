import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import joblib

# STEP 1: Load your CSV
df = pd.read_csv("Soil_HSV_Score_Model_Comparison.csv")
print("Available columns:", df.columns.tolist())

# STEP 2: Extract features and correct target column
X = df[["Hue (H)", "Saturation (S)", "Value (V)"]]       # Features
y = df["True Soil Score"]                                # ✅ Use correct column name

# STEP 3: Split for validation (optional)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# STEP 4: Train Random Forest
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# STEP 5: Save model
joblib.dump(model, "RandomForest_HSV_SoilScore_Model.pkl")
print("✅ Model saved as RandomForest_HSV_SoilScore_Model.pkl")

# STEP 6: Evaluate (optional)
r2 = model.score(X_test, y_test)
print(f"📊 R² score on test set: {r2:.4f}")
