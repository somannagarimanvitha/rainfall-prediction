import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Load dataset
data = pd.read_csv("Rainfall.csv")

# Clean columns
data.columns = data.columns.str.strip().str.lower()

# Encode target
data['rainfall'] = data['rainfall'].map({'yes':1,'no':0})

# Select ONLY features used in Streamlit
X = data[['humidity','pressure','windspeed','winddirection']]
y = data['rainfall']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

# Save correctly
joblib.dump(model, "model.pkl")

print("✅ model.pkl saved successfully")